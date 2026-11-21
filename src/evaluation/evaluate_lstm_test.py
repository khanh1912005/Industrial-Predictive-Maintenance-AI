
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

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "cmapss"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
SCALER_DIR = ARTIFACT_DIR / "scalers"
REPORT_DIR = BASE_DIR / "reports"

TEST_PATH = RAW_DIR / "test_FD001.txt"
RUL_PATH = RAW_DIR / "RUL_FD001.txt"

METADATA_PATH = SCALER_DIR / "feature_metadata.json"
SCALER_PATH = SCALER_DIR / "standard_scaler.joblib"
MODEL_PATH = MODEL_DIR / "lstm.pt"

PREDICTIONS_PATH = REPORT_DIR / "lstm_test_predictions.csv"
METRICS_PATH = REPORT_DIR / "lstm_test_metrics.csv"

SEQUENCE_LENGTH = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. LOAD FEATURE METADATA
# ==========================================

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

feature_columns = metadata["selected_features"]

print("\n===== SELECTED FEATURES =====")
print("Number of features:", len(feature_columns))
print("Features:", feature_columns)


# ==========================================
# 3. LOAD STANDARD SCALER
# ==========================================

scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(feature_columns):
    raise ValueError(
        "Scaler feature count does not match selected features."
    )


# ==========================================
# 4. LOAD NASA OFFICIAL TEST DATA
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

actual_rul = rul_df["RUL"].to_numpy(dtype=np.float32)

if len(unit_ids) != len(actual_rul):
    raise ValueError(
        "Number of test engines does not match RUL labels."
    )

if unit_ids != list(range(1, len(unit_ids) + 1)):
    raise ValueError(
        "Test engine IDs are not in the expected order."
    )

print("\n===== OFFICIAL TEST DATA =====")
print("Number of engines:", len(unit_ids))
print("Raw test rows:", len(test_df))


# ==========================================
# 5. BUILD LAST-30-CYCLE SEQUENCES
# ==========================================

X_test_sequences = []
final_cycles = []
start_cycles = []

for unit_id, engine_df in test_df.groupby(
    "unit_number",
    sort=True
):

    engine_df = engine_df.sort_values("time_cycles")

    # Take the last 30 observed cycles
    last_window = engine_df.tail(SEQUENCE_LENGTH)

    if len(last_window) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Engine {unit_id} has fewer than "
            f"{SEQUENCE_LENGTH} observed cycles."
        )

    # Keep exactly the 17 selected features
    features = last_window[feature_columns].to_numpy(
        dtype=np.float32
    )

    # Apply the scaler fitted on training data
    scaled_features = scaler.transform(features)

    X_test_sequences.append(scaled_features)

    start_cycles.append(
        int(last_window["time_cycles"].iloc[0])
    )

    final_cycles.append(
        int(last_window["time_cycles"].iloc[-1])
    )


X_test = np.stack(X_test_sequences).astype(np.float32)

print("\n===== LSTM TEST INPUT =====")
print("X_test shape:", X_test.shape)

if X_test.shape[1:] != (SEQUENCE_LENGTH, len(feature_columns)):
    raise ValueError("Unexpected LSTM test input shape.")

if not np.isfinite(X_test).all():
    raise ValueError("Test input contains NaN or Inf values.")


# ==========================================
# 6. LOAD TRAINED LSTM CHECKPOINT
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

input_size = int(checkpoint["input_size"])
hidden_size = int(checkpoint["hidden_size"])

if input_size != len(feature_columns):
    raise ValueError(
        "Checkpoint input size does not match feature count."
    )

model = LSTMRegressor(
    input_size=input_size,
    hidden_size=hidden_size
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("\n===== LOADED LSTM =====")
print("Hidden size:", hidden_size)
print("Best epoch:", checkpoint["best_epoch"])
print("Validation MAE:", checkpoint["validation_mae"])


# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)

with torch.no_grad():

    predictions = model(X_test_tensor)

predictions = predictions.cpu().numpy()

if not np.isfinite(predictions).all():
    raise ValueError("Model predictions contain NaN or Inf.")


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


# PHM08 asymmetric score
score_per_engine = np.where(
    errors < 0,
    np.expm1(-errors / 13),
    np.expm1(errors / 10)
)

phm08_score = score_per_engine.sum()

mean_error = errors.mean()

overprediction = int(np.sum(errors > 0))
underprediction = int(np.sum(errors < 0))
exact_prediction = int(np.sum(errors == 0))


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
    "Exact_Prediction": exact_prediction,
    "Negative_Predictions": int(np.sum(predictions < 0))
}])


REPORT_DIR.mkdir(parents=True, exist_ok=True)

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ==========================================
# 11. PRINT RESULTS
# ==========================================

print("\n===== LSTM OFFICIAL TEST METRICS =====")

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"PHM08 Score: {phm08_score:.4f}")

print("Mean Error:", f"{mean_error:.4f}")
print("Overprediction:", overprediction)
print("Underprediction:", underprediction)
print("Exact Prediction:", exact_prediction)
print("Negative Predictions:", int(np.sum(predictions < 0)))


print("\n===== SAMPLE PREDICTIONS =====")
print(prediction_df.head(10).to_string(index=False))

print("\nPredictions saved:", PREDICTIONS_PATH)
print("Metrics saved:", METRICS_PATH)
