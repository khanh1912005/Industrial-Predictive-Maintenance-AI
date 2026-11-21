
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD001.txt"
)

RUL_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "RUL_FD001.txt"
)

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "xgboost.joblib"
)

SCALER_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "standard_scaler.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

REPORT_DIR = BASE_DIR / "reports"

RUL_CAP = 125


def main():

    # ==========================================
    # 2. LOAD MODEL, SCALER AND METADATA
    # ==========================================

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    # ==========================================
    # 3. LOAD NASA TEST DATA
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

    print("===== TEST DATA INFORMATION =====")
    print("Test shape:", test_df.shape)
    print("Number of engines:", test_df["unit_number"].nunique())
    print("RUL labels:", len(rul_df))

    # ==========================================
    # 4. GET FINAL CYCLE OF EACH ENGINE
    # ==========================================

    test_final = (
        test_df
        .sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )

    if len(test_final) != len(rul_df):
        raise ValueError(
            "Number of test engines does not match RUL labels!"
        )

    expected_engine_ids = list(range(1, len(rul_df) + 1))

    actual_engine_ids = (
        test_final["unit_number"].astype(int).tolist()
    )

    if actual_engine_ids != expected_engine_ids:
        raise ValueError(
            "Engine order does not match NASA RUL file!"
        )

    print("\n===== FINAL CYCLE INFORMATION =====")
    print("Final engine data shape:", test_final.shape)
    print(test_final[["unit_number", "time_cycles"]].head())

    # ==========================================
    # 5. SELECT AND SCALE FEATURES
    # ==========================================

    X_test_raw = test_final[feature_columns]

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_columns,
    )

    # ==========================================
    # 6. PREDICT RUL
    # ==========================================

    predictions = model.predict(X_test_scaled)

    # ==========================================
    # 7. LOAD ACTUAL RUL
    # ==========================================

    y_test_raw = rul_df["RUL"].to_numpy()

    # Also evaluate against the same RUL cap
    # used during model training.
    y_test_capped = np.minimum(y_test_raw, RUL_CAP)

    # ==========================================
    # 8. EVALUATE MODEL
    # ==========================================

    metric_rows = []

    evaluation_targets = [
        ("NASA official RUL", y_test_raw),
        ("RUL capped at 125", y_test_capped),
    ]

    for evaluation_name, y_true in evaluation_targets:

        mae = mean_absolute_error(y_true, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_true, predictions)
        )

        r2 = r2_score(y_true, predictions)

        metric_rows.append({
            "evaluation": evaluation_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    metrics_df = pd.DataFrame(metric_rows)

    print("\n===== XGBOOST TEST METRICS =====")
    print(metrics_df.to_string(index=False))

    # ==========================================
    # 9. SAVE PREDICTIONS AND METRICS
    # ==========================================

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        "unit_number": test_final["unit_number"],
        "final_cycle": test_final["time_cycles"],
        "actual_RUL": y_test_raw,
        "actual_RUL_capped_125": y_test_capped,
        "predicted_RUL": predictions,
    })
        # ==========================================
    # 10. ERROR ANALYSIS
    # ==========================================

    # Calculate prediction errors
    predictions_df["error"] = (
        predictions_df["predicted_RUL"]
        - predictions_df["actual_RUL"]
    )

    # Calculate absolute errors
    predictions_df["absolute_error"] = (
        predictions_df["error"].abs()
    )

    # Count prediction directions
    overprediction = (predictions_df["error"] > 0).sum()
    underprediction = (predictions_df["error"] < 0).sum()
    exact_prediction = (predictions_df["error"] == 0).sum()

    print("\n===== ERROR ANALYSIS =====")
    print("Overprediction:", overprediction)
    print("Underprediction:", underprediction)
    print("Exact prediction:", exact_prediction)

    print(
        "Mean Error:",
        predictions_df["error"].mean()
    )

    # Find 10 engines with largest errors
    worst_predictions = predictions_df.sort_values(
        by="absolute_error",
        ascending=False,
    ).head(10)

    print("\n===== TOP 10 WORST PREDICTIONS =====")
    print(worst_predictions.to_string(index=False))

    # ==========================================
    # 11. EXPLAIN ENGINE 41 USING SHAP
    # ==========================================

    target_engine = 41

    # Find Engine 41 position in Test data
    engine_position = test_final.index[
        test_final["unit_number"] == target_engine
    ][0]

    # Select this engine's features
    engine_features = X_test_scaled.iloc[[engine_position]]

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    engine_shap_values = explainer(engine_features)

    # Actual and predicted RUL
    engine_result = predictions_df[
        predictions_df["unit_number"] == target_engine
    ].iloc[0]

    print("\n===== ENGINE 41 SHAP EXPLANATION =====")

    print("Actual RUL:", engine_result["actual_RUL"])
    print("Predicted RUL:", engine_result["predicted_RUL"])
    print("Prediction Error:", engine_result["error"])

    # Create feature contribution table
    engine_shap = pd.DataFrame({
        "feature": feature_columns,
        "raw_feature_value": X_test_raw.iloc[engine_position].values,
        "scaled_feature_value": X_test_scaled.iloc[engine_position].values,
        "shap_value": engine_shap_values.values[0],
    })

    engine_shap["absolute_shap"] = (
        engine_shap["shap_value"].abs()
    )

    engine_shap = engine_shap.sort_values(
        by="absolute_shap",
        ascending=False,
    )

    print("\n===== ENGINE 41 FEATURE CONTRIBUTIONS =====")
    print(engine_shap.to_string(index=False))

    # Verify SHAP additive explanation
    base_value = float(
        np.asarray(engine_shap_values.base_values).reshape(-1)[0]
    )

    # ==========================================
    # 12. SHAP WATERFALL PLOT - ENGINE 41
    # ==========================================

    # Clear previous matplotlib figures
    plt.close("all")

    # Create SHAP Waterfall Plot
    shap.plots.waterfall(
        engine_shap_values[0],
        max_display=17,
        show=False,
    )

    # Adjust chart layout
    plt.tight_layout()

    # Save chart
    waterfall_path = (
        REPORT_DIR / "engine_41_shap_waterfall.png"
    )

    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nWaterfall plot saved:", waterfall_path)

    # Display chart
    plt.show()
    plt.close()


    print("\nBase value:", base_value)
    print(
        "Base + SHAP contributions:",
        base_value + engine_shap_values.values[0].sum()
    )

    predictions_df.to_csv(
        REPORT_DIR / "xgboost_test_predictions.csv",
        index=False,
    )

    metrics_df.to_csv(
        REPORT_DIR / "xgboost_test_metrics.csv",
        index=False,
    )

    print("\n===== SAMPLE PREDICTIONS =====")
    print(predictions_df.head(10).to_string(index=False))

    print("\nTest evaluation completed successfully!")


if __name__ == "__main__":
    main()
