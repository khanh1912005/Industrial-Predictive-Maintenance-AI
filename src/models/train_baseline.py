
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

TRAIN_FILE = (
    DATA_DIR / "train_FD001_standard_scaled.csv"
)

VAL_FILE = (
    DATA_DIR / "validation_FD001_standard_scaled.csv"
)


# ==========================================
# 2. LOAD FEATURE METADATA
# ==========================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

FEATURE_COLUMNS = metadata["selected_features"]


# ==========================================
# 3. LOAD DATA
# ==========================================

print("\n===== LOAD TRAINING DATA =====")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["RUL"]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df["RUL"]

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

print(f"Number of features: {len(FEATURE_COLUMNS)}")


# ==========================================
# 4. VERIFY DATA
# ==========================================

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain NaN or infinity.")

if not np.isfinite(X_val.to_numpy()).all():
    raise ValueError("Validation features contain NaN or infinity.")

if not np.isfinite(y_train.to_numpy()).all():
    raise ValueError("Training target contains NaN or infinity.")

if not np.isfinite(y_val.to_numpy()).all():
    raise ValueError("Validation target contains NaN or infinity.")

train_engines = set(train_df["unit_number"])
val_engines = set(val_df["unit_number"])

if train_engines.intersection(val_engines):
    raise ValueError("Engine overlap detected.")


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Dummy Regressor": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression": LinearRegression()

}


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

results = []

validation_predictions = val_df[
    ["unit_number", "time_cycles", "RUL"]
].copy()

validation_predictions = validation_predictions.rename(
    columns={"RUL": "RUL_actual"}
)

for model_name, model in models.items():

    print(f"\n===== TRAINING: {model_name} =====")

    # Train model
    model.fit(X_train, y_train)

    # Predict on validation
    y_pred = model.predict(X_val)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_val, y_pred)
    )

    r2 = r2_score(y_val, y_pred)

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # Save predictions
    prediction_column = (
        model_name.lower().replace(" ", "_")
        + "_prediction"
    )

    validation_predictions[prediction_column] = y_pred


    # Save trained model
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        ARTIFACT_DIR
        / f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")


# ==========================================
# 7. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE",
    ascending=True
)

print("\n===== MODEL COMPARISON =====")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    REPORT_DIR / "baseline_model_results.csv",
    index=False
)

validation_predictions.to_csv(
    REPORT_DIR / "baseline_validation_predictions.csv",
    index=False
)

print("\n===== BASELINE COMPLETED =====")
