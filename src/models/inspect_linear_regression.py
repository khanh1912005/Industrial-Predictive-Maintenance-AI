
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "linear_regression.joblib"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_FD001_standard_scaled.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"


# ==========================================
# 2. CHECK FILES
# ==========================================

for file_path in [
    MODEL_FILE,
    METADATA_FILE,
    VAL_FILE
]:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}"
        )


# ==========================================
# 3. LOAD MODEL AND FEATURES
# ==========================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

FEATURE_COLUMNS = metadata["selected_features"]

model = joblib.load(MODEL_FILE)

if len(model.coef_) != len(FEATURE_COLUMNS):
    raise ValueError(
        "So luong weights khong khop voi features."
    )


# ==========================================
# 4. INSPECT BIAS
# ==========================================

print("\n===== MODEL BIAS =====")

print(f"Bias: {model.intercept_:.4f}")


# ==========================================
# 5. INSPECT WEIGHTS
# ==========================================

print("\n===== MODEL WEIGHTS =====")

weights_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "weight": model.coef_
})

weights_df["absolute_weight"] = (
    weights_df["weight"].abs()
)

weights_df = weights_df.sort_values(
    "absolute_weight",
    ascending=False
)

print(
    weights_df.round(4).to_string(index=False)
)


# ==========================================
# 6. VALIDATION PREDICTIONS
# ==========================================

print("\n===== PREDICTION INSPECTION =====")

val_df = pd.read_csv(VAL_FILE)

X_val = val_df[FEATURE_COLUMNS]

y_actual = val_df["RUL"]

y_pred = model.predict(X_val)

prediction_df = val_df[
    ["unit_number", "time_cycles", "RUL"]
].copy()

prediction_df = prediction_df.rename(
    columns={"RUL": "RUL_actual"}
)

prediction_df["RUL_predicted"] = y_pred

prediction_df["error"] = (
    prediction_df["RUL_predicted"]
    - prediction_df["RUL_actual"]
)

prediction_df["absolute_error"] = (
    prediction_df["error"].abs()
)

print(
    prediction_df.head(10).round(2).to_string(index=False)
)

print(
    f"\nValidation MAE: "
    f"{mean_absolute_error(y_actual, y_pred):.4f}"
)


# ==========================================
# 7. SAVE WEIGHT REPORT
# ==========================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

weights_df.to_csv(
    REPORT_DIR / "linear_regression_coefficients.csv",
    index=False
)

print("\n===== INSPECTION COMPLETED =====")
print("Weight report saved successfully.")
