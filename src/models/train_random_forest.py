
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_FD001_standard_scaled.csv"
)

VALIDATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation_FD001_standard_scaled.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"


# ==========================================
# 2. LOAD DATA
# ==========================================

def main():

    # Kiểm tra file đầu vào
    required_files = [
        TRAIN_PATH,
        VALIDATION_PATH,
        METADATA_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {file_path}"
            )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    if "RUL" not in train_df.columns:
        raise ValueError("Training dataset thiếu cột RUL.")

    if "RUL" not in validation_df.columns:
        raise ValueError("Validation dataset thiếu cột RUL.")

    missing_train = set(feature_columns) - set(train_df.columns)
    missing_val = set(feature_columns) - set(validation_df.columns)

    if missing_train or missing_val:
        raise ValueError(
            f"Feature bị thiếu. Train: {missing_train}, "
            f"Validation: {missing_val}"
        )

    print("===== DATA INFORMATION =====")
    print(f"Training shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Number of features: {len(feature_columns)}")


    # ==========================================
    # 3. SEPARATE FEATURES AND TARGET
    # ==========================================

    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["RUL"]


    # ==========================================
    # 4. BUILD RANDOM FOREST MODEL
    # ==========================================

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


    # ==========================================
    # 5. TRAIN MODEL
    # ==========================================

    print("\n===== MODEL TRAINING =====")

    model.fit(X_train, y_train)

    print("Random Forest training completed!")


    # ==========================================
    # 6. MAKE PREDICTIONS
    # ==========================================

    y_pred = model.predict(X_validation)


    # ==========================================
    # 7. EVALUATE MODEL
    # ==========================================

    mae = mean_absolute_error(y_validation, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_validation, y_pred)
    )

    r2 = r2_score(y_validation, y_pred)

    print("\n===== RANDOM FOREST RESULTS =====")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")


    # ==========================================
    # 8. SAVE MODEL AND REPORTS
    # ==========================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "random_forest.joblib"
    joblib.dump(model, model_path)

    metrics = pd.DataFrame([{
        "model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }])

    metrics_path = REPORT_DIR / "random_forest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Lưu mức độ quan trọng của từng feature
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    importance_path = (
        REPORT_DIR / "random_forest_feature_importance.csv"
    )

    importance.to_csv(importance_path, index=False)

    print("\n===== FEATURE IMPORTANCE =====")
    print(importance.to_string(index=False))

    print("\n===== SAVED FILES =====")
    print(model_path)
    print(metrics_path)
    print(importance_path)


if __name__ == "__main__":
    main()
