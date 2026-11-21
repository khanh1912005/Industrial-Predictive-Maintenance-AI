
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

PACKAGE_DIR = BASE_DIR / "artifacts" / "final_dl_model"
RAW_DIR = BASE_DIR / "data" / "raw" / "cmapss"

MODEL_PATH = PACKAGE_DIR / "lstm.pt"
SCALER_PATH = PACKAGE_DIR / "standard_scaler.joblib"
METADATA_PATH = PACKAGE_DIR / "feature_metadata.json"
CONFIG_PATH = PACKAGE_DIR / "model_config.json"

TEST_PATH = RAW_DIR / "test_FD001.txt"

SEQUENCE_LENGTH = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. VERIFY PACKAGE FILES
# ==========================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    CONFIG_PATH
]

for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Package file not found: {file_path}"
        )

print("All package files found.")


# ==========================================
# 3. LOAD CONFIGURATION AND METADATA
# ==========================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


feature_columns = metadata["selected_features"]

input_size = int(config["input_size"])
hidden_size = int(config["hidden_size"])

sequence_length = int(config["sequence_length"])

if len(feature_columns) != input_size:
    raise ValueError(
        "Feature count does not match model input size."
    )

if sequence_length != SEQUENCE_LENGTH:
    raise ValueError(
        "Unexpected sequence length."
    )


# ==========================================
# 4. LOAD PACKAGED SCALER
# ==========================================

scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(feature_columns):
    raise ValueError(
        "Scaler feature count does not match metadata."
    )


# ==========================================
# 5. LOAD PACKAGED MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = LSTMRegressor(
    input_size=input_size,
    hidden_size=hidden_size
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\n===== PACKAGE INFORMATION =====")
print("Model:", config["model_name"])
print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Sequence length:", sequence_length)
print("Feature count:", len(feature_columns))
print("Scaler:", config["scaler_type"])


# ==========================================
# 6. LOAD A SENSOR SEQUENCE
# ==========================================

raw_columns = (
    ["unit_number", "time_cycles"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    TEST_PATH,
    sep=r"\s+",
    header=None,
    names=raw_columns
)

test_df = test_df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)


# Use Engine 1 as an inference example
engine_id = 1

engine_df = test_df[
    test_df["unit_number"] == engine_id
].sort_values("time_cycles")

last_window = engine_df.tail(
    sequence_length
)

if len(last_window) != sequence_length:
    raise ValueError(
        "Not enough cycles for the inference window."
    )


# ==========================================
# 7. PREPROCESS SENSOR SEQUENCE
# ==========================================

raw_features = last_window[
    feature_columns
].to_numpy(dtype=np.float32)

if not np.isfinite(raw_features).all():
    raise ValueError(
        "Raw sensor sequence contains NaN or Inf."
    )

raw_features_df = pd.DataFrame(
    raw_features,
    columns=feature_columns
)

scaled_features = scaler.transform(
    raw_features_df
).astype(np.float32)

if not np.isfinite(scaled_features).all():
    raise ValueError(
        "Scaled sensor sequence contains NaN or Inf."
    )


# Add batch dimension
input_tensor = torch.from_numpy(
    scaled_features
).unsqueeze(0).to(DEVICE)

print("\n===== INFERENCE INPUT =====")
print("Engine:", engine_id)
print(
    "Observed cycles:",
    int(last_window["time_cycles"].iloc[0]),
    "->",
    int(last_window["time_cycles"].iloc[-1])
)
print("Input shape:", tuple(input_tensor.shape))


# ==========================================
# 8. GENERATE RUL PREDICTION
# ==========================================

with torch.no_grad():

    prediction = model(input_tensor)

predicted_rul = float(
    prediction.item()
)

if not np.isfinite(predicted_rul):
    raise ValueError(
        "Model returned a non-finite prediction."
    )


# ==========================================
# 9. DISPLAY INFERENCE RESULT
# ==========================================

print("\n===== FINAL DL MODEL SMOKE TEST =====")

print("Predicted RUL:", round(predicted_rul, 4))
print("Prediction is finite:", np.isfinite(predicted_rul))
print("Model evaluation mode:", not model.training)

print("\nPackage inference test completed successfully.")
    