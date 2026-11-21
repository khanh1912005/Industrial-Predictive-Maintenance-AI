
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "raw" / "cmapss"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
SCALER_DIR = BASE_DIR / "artifacts" / "scalers"
REPORT_DIR = BASE_DIR / "reports"

TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"

SCALER_PATH = SCALER_DIR / "standard_scaler.joblib"
METADATA_PATH = SCALER_DIR / "feature_metadata.json"

MODEL_PATHS = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}


def main():

    # ==========================================
    # 2. LOAD METADATA AND SCALER
    # ==========================================

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    scaler = joblib.load(SCALER_PATH)

    # ==========================================
    # 3. LOAD ALL TRAINED MODELS
    # ==========================================

    models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    # ==========================================
    # 4. LOAD NASA TEST DATA
    # ==========================================

    columns = (
        [
            "unit_number",
            "time_cycles",
            "setting_1",
            "setting_2",
            "setting_3",
        ]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    test_df = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    rul_df = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    # ==========================================
    # 5. GET FINAL CYCLE OF EACH ENGINE
    # ==========================================

    test_final = (
        test_df
        .sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )

    y_test = rul_df["RUL"].to_numpy()

    if len(test_final) != len(y_test):
        raise ValueError(
            "Test engine count does not match RUL labels!"
        )

    print("Number of test engines:", len(test_final))

    # ==========================================
    # 6. PREPARE TEST FEATURES
    # ==========================================

    X_test_raw = test_final[feature_columns]

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_columns,
    )

    # ==========================================
    # 7. EVALUATE ALL MODELS
    # ==========================================

    metrics_results = []

    predictions_df = pd.DataFrame({
        "unit_number": test_final["unit_number"],
        "final_cycle": test_final["time_cycles"],
        "actual_RUL": y_test,
    })

    for model_name, model in models.items():

        predictions = model.predict(X_test_scaled)

        errors = predictions - y_test

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        mean_error = errors.mean()

        overprediction = (errors > 0).sum()
        underprediction = (errors < 0).sum()

        metrics_results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Mean_Error": mean_error,
            "Overprediction": int(overprediction),
            "Underprediction": int(underprediction),
        })

        predictions_df[
            f"{model_name}_predicted_RUL"
        ] = predictions

    # ==========================================
    # 8. DISPLAY COMPARISON
    # ==========================================

    metrics_df = pd.DataFrame(metrics_results)

    metrics_df = metrics_df.sort_values(
        by="MAE",
        ascending=True,
    )

    print("\n===== ML MODELS OFFICIAL TEST COMPARISON =====")

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==========================================
    # 9. SAVE REPORTS
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        REPORT_DIR / "ml_models_test_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        REPORT_DIR / "ml_models_test_predictions.csv",
        index=False,
    )

    print("\nComparison reports saved successfully!")


if __name__ == "__main__":
    main()
