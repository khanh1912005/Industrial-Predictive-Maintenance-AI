
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

MODEL_PATHS = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}

# Target RUL levels for pseudo-test
TARGET_RUL_LEVELS = [20, 40, 60, 80, 100, 120]


# ==========================================
# 2. PHM08 SCORING FUNCTION
# ==========================================

def calculate_phm08_score(y_true, y_pred):

    errors = y_pred - y_true

    penalties = np.zeros_like(errors, dtype=float)

    under_mask = errors < 0
    over_mask = errors >= 0

    penalties[under_mask] = np.expm1(
        -errors[under_mask] / 13.0
    )

    penalties[over_mask] = np.expm1(
        errors[over_mask] / 10.0
    )

    return penalties


# ==========================================
# 3. METRICS FUNCTION
# ==========================================


def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true

    penalties = calculate_phm08_score(
        y_true,
        y_pred,
    )

    # R2 is undefined when actual values are constant
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)

    return {
        "n_samples": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2,
        "Mean_Error": errors.mean(),
        "PHM08_Total_Score": penalties.sum(),
        "PHM08_Mean_Score": penalties.mean(),
        "Overprediction": int((errors > 0).sum()),
        "Underprediction": int((errors < 0).sum()),
    }


# ==========================================
# 4. MAIN FUNCTION
# ==========================================

def main():

    # ==========================================
    # 4.1 LOAD FEATURE METADATA
    # ==========================================

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    # ==========================================
    # 4.2 LOAD VALIDATION DATA
    # ==========================================

    validation_df = pd.read_csv(VALIDATION_PATH)

    required_columns = [
        "unit_number",
        "time_cycles",
        "RUL",
        *feature_columns,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in validation_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("===== VALIDATION DATA INFORMATION =====")
    print("Rows:", len(validation_df))
    print("Engines:", validation_df["unit_number"].nunique())

    # ==========================================
    # 4.3 CREATE PSEUDO-TEST SAMPLES
    # ==========================================

    selected_rows = []
    missing_targets = []

    for engine_id, engine_df in validation_df.groupby(
        "unit_number",
        sort=True,
    ):

        engine_df = engine_df.sort_values("time_cycles")

        for target_rul in TARGET_RUL_LEVELS:

            # Find the row whose actual RUL
            # matches the target RUL level.

            mask = np.isclose(
                engine_df["RUL"].to_numpy(),
                target_rul,
            )

            candidates = engine_df.loc[mask]

            if candidates.empty:

                missing_targets.append({
                    "unit_number": engine_id,
                    "target_RUL": target_rul,
                })

                continue

            # Select the matching observation
            selected_row = candidates.iloc[0].copy()

            selected_row["target_rul_level"] = target_rul

            selected_rows.append(selected_row)

    if not selected_rows:
        raise ValueError(
            "No pseudo-test samples could be created!"
        )

    pseudo_test = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    # ==========================================
    # 4.4 CHECK SAMPLE COVERAGE
    # ==========================================

    print("\n===== PSEUDO-TEST COVERAGE =====")

    coverage = (
        pseudo_test
        .groupby("target_rul_level")["unit_number"]
        .nunique()
        .reindex(TARGET_RUL_LEVELS, fill_value=0)
    )

    print(coverage.to_string())

    print("\nTotal pseudo-test samples:", len(pseudo_test))

    print(
        "Engines represented:",
        pseudo_test["unit_number"].nunique(),
    )

    print(
        "Missing engine/target combinations:",
        len(missing_targets),
    )

    if missing_targets:
        print("Some engines do not have every target RUL level.")

    # ==========================================
    # 4.5 PREPARE FEATURES AND TARGETS
    # ==========================================

    X_pseudo = pseudo_test[feature_columns]

    y_pseudo = pseudo_test["RUL"].to_numpy()

    # ==========================================
    # 4.6 LOAD TRAINED MODELS
    # ==========================================

    models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    # ==========================================
    # 4.7 EVALUATE MODELS
    # ==========================================

    overall_results = []
    group_results = []

    prediction_report = pseudo_test[
        [
            "unit_number",
            "time_cycles",
            "RUL",
            "target_rul_level",
        ]
    ].copy()

    for model_name, model in models.items():

        # Generate predictions for all pseudo-test samples
        predictions = model.predict(X_pseudo)

        # Save predictions for this model
        prediction_report[
            f"{model_name}_predicted_RUL"
        ] = predictions

        # --------------------------------------
        # Overall metrics
        # --------------------------------------

        overall_metrics = calculate_metrics(
            y_pseudo,
            predictions,
        )

        overall_metrics["model"] = model_name

        overall_results.append(overall_metrics)

        # --------------------------------------
        # Metrics by actual RUL level
        # --------------------------------------

        for target_rul, group_df in pseudo_test.groupby(
            "target_rul_level",
            sort=True,
        ):

            positions = group_df.index.to_numpy()

            group_true = y_pseudo[positions]
            group_pred = predictions[positions]

            group_metrics = calculate_metrics(
                group_true,
                group_pred,
            )

            group_metrics["model"] = model_name
            group_metrics["target_RUL"] = target_rul

            group_results.append(group_metrics)

    # ==========================================
    # 4.8 CREATE RESULT TABLES
    # ==========================================

    overall_df = pd.DataFrame(overall_results)

    by_group_df = pd.DataFrame(group_results)

    print("\n===== PSEUDO-TEST OVERALL COMPARISON =====")

    print(
        overall_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n===== PSEUDO-TEST PERFORMANCE BY RUL =====")

    print(
        by_group_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==========================================
    # 4.9 SAVE REPORTS
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        REPORT_DIR / "ml_pseudo_test_overall_metrics.csv",
        index=False,
    )

    by_group_df.to_csv(
        REPORT_DIR / "ml_pseudo_test_metrics_by_rul.csv",
        index=False,
    )

    prediction_report.to_csv(
        REPORT_DIR / "ml_pseudo_test_predictions.csv",
        index=False,
    )

    print("\nPseudo-test evaluation completed!")
    print("Reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
