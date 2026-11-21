
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


# ==========================================
# 2. PHM08 SCORING FUNCTION
# ==========================================

def calculate_phm08_score(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true

    penalties = np.zeros_like(errors)

    # Underprediction: predicted RUL < actual RUL
    under_mask = errors < 0

    penalties[under_mask] = np.expm1(
        -errors[under_mask] / 13.0
    )

    # Overprediction: predicted RUL > actual RUL
    over_mask = errors >= 0

    penalties[over_mask] = np.expm1(
        errors[over_mask] / 10.0
    )

    return penalties


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    # ==========================================
    # 3.1 LOAD FEATURE METADATA
    # ==========================================

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    # ==========================================
    # 3.2 LOAD VALIDATION DATA
    # ==========================================

    validation_df = pd.read_csv(VALIDATION_PATH)

    X_validation = validation_df[feature_columns]

    y_validation = validation_df["RUL"].to_numpy()

    print("===== VALIDATION INFORMATION =====")
    print("Number of rows:", len(validation_df))
    print("Number of features:", len(feature_columns))

    # ==========================================
    # 3.3 LOAD TRAINED MODELS
    # ==========================================

    models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    # ==========================================
    # 3.4 EVALUATE MODELS
    # ==========================================

    results = []

    for model_name, model in models.items():

        # Generate predictions
        predictions = model.predict(X_validation)

        # Calculate prediction errors
        errors = predictions - y_validation

        # --------------------------------------
        # Standard Regression Metrics
        # --------------------------------------

        mae = mean_absolute_error(
            y_validation,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_validation,
                predictions,
            )
        )

        r2 = r2_score(
            y_validation,
            predictions,
        )

        # --------------------------------------
        # PHM08 Asymmetric Score
        # --------------------------------------

        penalties = calculate_phm08_score(
            y_validation,
            predictions,
        )

        total_score = penalties.sum()

        mean_score = penalties.mean()

        # --------------------------------------
        # Separate Error Directions
        # --------------------------------------

        over_mask = errors > 0

        under_mask = errors < 0

        overprediction = over_mask.sum()

        underprediction = under_mask.sum()

        # --------------------------------------
        # Separate PHM08 Penalties
        # --------------------------------------

        over_penalty = penalties[over_mask].sum()

        under_penalty = penalties[under_mask].sum()

        # --------------------------------------
        # Calculate Penalty Percentages
        # --------------------------------------

        if total_score>0:
            over_share = (
                over_penalty / total_score * 100
            )

            under_share = (
                under_penalty / total_score * 100
            )

        else:

            over_share = 0

            under_share = 0

        # --------------------------------------
        # Save ONE Result Per Model
        # --------------------------------------

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "PHM08_Total_Score": total_score,
            "PHM08_Mean_Score": mean_score,
            "Over_Penalty": over_penalty,
            "Under_Penalty": under_penalty,
            "Over_Penalty_Share_%": over_share,
            "Under_Penalty_Share_%": under_share,
            "Overprediction": int(overprediction),
            "Underprediction": int(underprediction),
        })

    # ==========================================
    # 3.5 CREATE FINAL COMPARISON TABLE
    # ==========================================

    results_df = pd.DataFrame(results)

    # Sort models by PHM08 Total Score
    # Lower score is better

    results_df = results_df.sort_values(
        by="PHM08_Total_Score",
        ascending=True,
    ).reset_index(drop=True)

    print("\n===== VALIDATION MODEL SELECTION =====")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==========================================
    # 3.6 SAVE FINAL COMPARISON REPORT
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR
        / "ml_models_validation_selection.csv"
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    print("\nValidation comparison saved:", report_path)

    print("\nNumber of models evaluated:", len(results_df))

    print("\nModel evaluation completed successfully!")


# ==========================================
# 4. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
