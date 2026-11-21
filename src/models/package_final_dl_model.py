
import json
import shutil
from pathlib import Path

import joblib
import torch

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "artifacts" / "models" / "lstm.pt"

SCALER_PATH = (
    BASE_DIR / "artifacts" / "scalers" / "standard_scaler.joblib"
)

METADATA_PATH = (
    BASE_DIR / "artifacts" / "scalers" / "feature_metadata.json"
)

PACKAGE_DIR = (
    BASE_DIR / "artifacts" / "final_dl_model"
)


# ==========================================
# 2. CHECK SOURCE FILES
# ==========================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH
]

for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ==========================================
# 3. LOAD FEATURE METADATA
# ==========================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

feature_columns = metadata["selected_features"]

if len(feature_columns) != 17:
    raise ValueError(
        "Expected exactly 17 selected features."
    )


# ==========================================
# 4. VERIFY SCALER
# ==========================================

scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(feature_columns):
    raise ValueError(
        "Scaler and feature metadata do not match."
    )


# ==========================================
# 5. LOAD AND VERIFY LSTM
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

input_size = int(checkpoint["input_size"])
hidden_size = int(checkpoint["hidden_size"])

if input_size != len(feature_columns):
    raise ValueError(
        "LSTM input size does not match feature count."
    )

model = LSTMRegressor(
    input_size=input_size,
    hidden_size=hidden_size
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# 6. CREATE PACKAGE DIRECTORY
# ==========================================

PACKAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 7. COPY MODEL AND PREPROCESSING FILES
# ==========================================

shutil.copy2(
    MODEL_PATH,
    PACKAGE_DIR / "lstm.pt"
)

shutil.copy2(
    SCALER_PATH,
    PACKAGE_DIR / "standard_scaler.joblib"
)

shutil.copy2(
    METADATA_PATH,
    PACKAGE_DIR / "feature_metadata.json"
)


# ==========================================
# 8. CREATE MODEL CONFIGURATION
# ==========================================

model_config = {
    "project": "Industrial Predictive Maintenance AI",
    "model_name": "LSTM-Capped",
    "architecture": "LSTMRegressor",
    "framework": "PyTorch",

    "input_size": input_size,
    "hidden_size": hidden_size,

    "sequence_length": 30,
    "feature_count": len(feature_columns),
    "feature_columns": feature_columns,

    "scaler_type": scaler.__class__.__name__,
    "scaler_file": "standard_scaler.joblib",

    "target": "RUL",
    "rul_cap": 125,

    "best_epoch": int(checkpoint["best_epoch"]),
    "validation_mae": float(
        checkpoint["validation_mae"]
    ),

    "model_file": "lstm.pt",
    "metadata_file": "feature_metadata.json"
}

CONFIG_PATH = PACKAGE_DIR / "model_config.json"

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_config,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# 9. VERIFY PACKAGE
# ==========================================

print("\n===== FINAL DL MODEL PACKAGE =====")

for file_path in PACKAGE_DIR.iterdir():

    if file_path.is_file():

        print(
            file_path.name,
            "|",
            file_path.stat().st_size,
            "bytes"
        )

print("\n===== MODEL INFORMATION =====")

print("Model:", model_config["model_name"])
print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Sequence length:", model_config["sequence_length"])
print("Features:", len(feature_columns))
print("Best epoch:", checkpoint["best_epoch"])
print("Validation MAE:", checkpoint["validation_mae"])

print("\nPackage successfully created:", PACKAGE_DIR)
