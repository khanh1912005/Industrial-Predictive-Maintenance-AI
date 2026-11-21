
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models.cnn_model import CNN1DRegressor


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "cmapss"
SCALER_DIR = BASE_DIR / "artifacts" / "scalers"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"

TEST_PATH = RAW_DIR / "test_FD001.txt"
RUL_PATH = RAW_DIR / "RUL_FD001.txt"

METADATA_PATH = SCALER_DIR / "feature_metadata.json"
SCALER_PATH = SCALER_DIR / "standard_scaler.joblib"

MODEL_PATH = MODEL_DIR / "cnn.pt"

PREDICTIONS_PATH = REPORT_DIR / "cnn_test_predictions.csv"
METRICS_PATH = REPORT_DIR / "cnn_test_metrics.csv"

SEQUENCE_LENGTH = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. LOAD FEATURE METADATA
# ==========================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

feature_columns = metadata["selected_features"]

print("\nNumber of features:", len(feature_columns))


# ==========================================
# 3. LOAD STANDARD SCALER
# ==========================================

scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(feature_columns):
    raise ValueError(
        "Scaler feature count does not match metadata."
    )


# ==========================================
# 4. LOAD OFFICIAL TEST DATA
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

rul_df = pd.read_csv(
    RUL_PATH,
    sep=r"\s+",
    header=None,
    names=["RUL"]
)

test_df = test_df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)

unit_ids = sorted(
    test_df["unit_number"].unique().tolist()
)

actual_rul = rul_df["RUL"].to_numpy(
    dtype=np.float32
)

if len(unit_ids) != len(actual_rul):
    raise ValueError(
        "Test engine count does not match RUL labels."
    )

if unit_ids != list(range(1, len(unit_ids) + 1)):
    raise ValueError(
        "Unexpected test engine IDs."
    )

print("\n===== OFFICIAL TEST DATA =====")
print("Number of engines:", len(unit_ids))
print("Raw test rows:", len(test_df))


# ==========================================
# 5. BUILD LAST-30-CYCLE SEQUENCES
# ==========================================

X_test_sequences = []
start_cycles = []
final_cycles = []

for unit_id, engine_df in test_df.groupby(
    "unit_number",
    sort=True
):

    engine_df = engine_df.sort_values(
        "time_cycles"
    )

    last_window = engine_df.tail(
        SEQUENCE_LENGTH
    )

    if len(last_window) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Engine {unit_id} has fewer than "
            f"{SEQUENCE_LENGTH} cycles."
        )

    features = last_window[
        feature_columns
    ].to_numpy(dtype=np.float32)

    scaled_features = scaler.transform(
        features
    )

    X_test_sequences.append(
        scaled_features
    )

    start_cycles.append(
        int(last_window["time_cycles"].iloc[0])
    )

    final_cycles.append(
        int(last_window["time_cycles"].iloc[-1])
    )


X_test = np.stack(
    X_test_sequences
).astype(np.float32)

print("\n===== TEST INPUT =====")
print("X_test shape:", X_test.shape)

if X_test.shape != (len(unit_ids), 30, len(feature_columns)):
    raise ValueError("Unexpected CNN test input shape.")

if not np.isfinite(X_test).all():
    raise ValueError("Test input contains NaN or Inf.")


# ==========================================
# 6. LOAD CNN CHECKPOINT
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

input_channels = int(
    checkpoint["input_channels"]
)

if input_channels != len(feature_columns):
    raise ValueError(
        "Checkpoint input channels do not match features."
    )

model = CNN1DRegressor(
    input_channels=input_channels
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\n===== LOADED CNN =====")
print("Input channels:", input_channels)
print("Best epoch:", checkpoint["best_epoch"])
print("Validation MAE:", checkpoint["validation_mae"])


# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

X_test_tensor = torch.from_numpy(
    X_test
).float().to(DEVICE)

with torch.no_grad():

    predictions = model(
        X_test_tensor
    )

predictions = predictions.cpu().numpy()

if not np.isfinite(predictions).all():
    raise ValueError("Predictions contain NaN or Inf.")


# ==========================================
# 8. CALCULATE OFFICIAL TEST METRICS
# ==========================================

errors = predictions - actual_rul

mae = mean_absolute_error(
    actual_rul,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(actual_rul, predictions)
)

r2 = r2_score(
    actual_rul,
    predictions
)

score_per_engine = np.where(
    errors < 0,
    np.expm1(-errors / 13),
    np.expm1(errors / 10)
)

phm08_score = score_per_engine.sum()

mean_error = errors.mean()

overprediction = int(
    np.sum(errors > 0)
)

underprediction = int(
    np.sum(errors < 0)
)

negative_predictions = int(
    np.sum(predictions < 0)
)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "unit_number": unit_ids,
    "start_cycle": start_cycles,
    "final_cycle": final_cycles,
    "actual_RUL": actual_rul,
    "predicted_RUL": predictions,
    "error": errors,
    "absolute_error": np.abs(errors)
})


# ==========================================
# 10. SAVE METRICS
# ==========================================

metrics_df = pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "PHM08_Score": phm08_score,
    "Mean_Error": mean_error,
    "Overprediction": overprediction,
    "Underprediction": underprediction,
    "Negative_Predictions": negative_predictions
}])


# ==========================================
# 11. SAVE REPORTS
# ==========================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n===== CNN OFFICIAL TEST METRICS =====")

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"PHM08 Score: {phm08_score:.4f}")

print(f"Mean Error: {mean_error:.4f}")
print("Overprediction:", overprediction)
print("Underprediction:", underprediction)
print("Negative Predictions:", negative_predictions)

print("\n===== SAMPLE PREDICTIONS =====")

print(
    prediction_df.head(10).to_string(index=False)
)

print("\nPredictions saved:", PREDICTIONS_PATH)
print("Metrics saved:", METRICS_PATH)
