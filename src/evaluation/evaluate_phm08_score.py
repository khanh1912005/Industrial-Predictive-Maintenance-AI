
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    BASE_DIR
    / "reports"
    / "ml_models_test_predictions.csv"
)

REPORT_DIR = BASE_DIR / "reports"


# ==========================================
# 2. PHM08 SCORING FUNCTION
# ==========================================

def calculate_phm08_score(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape!"
        )

    errors = y_pred - y_true

    penalties = np.zeros_like(errors)

    # Underprediction: predicted RUL is lower than actual RUL
    under_mask = errors < 0

    penalties[under_mask] = np.expm1(
        -errors[under_mask] / 13.0
    )

    # Overprediction: predicted RUL is higher than actual RUL
    over_mask = errors >= 0

    penalties[over_mask] = np.expm1(
        errors[over_mask] / 10.0
    )

    return errors, penalties


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    # ==========================================
    # 3.1 LOAD PREDICTIONS
    # ==========================================

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    # Model name and corresponding prediction column
    model_columns = {
        "Linear Regression": "Linear Regression_predicted_RUL",
        "Random Forest": "Random Forest_predicted_RUL",
        "XGBoost": "XGBoost_predicted_RUL",
    }

    required_columns = [
        "actual_RUL",
        *model_columns.values(),
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    y_true = df["actual_RUL"].to_numpy()

    # ==========================================
    # 3.2 EVALUATE EACH MODEL
    # ==========================================

    results = []

    for model_name, prediction_column in model_columns.items():

        y_pred = df[prediction_column].to_numpy()

        errors, penalties = calculate_phm08_score(
            y_true,
            y_pred,
        )

        # Standard regression metrics
        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        # PHM08 asymmetric score
        total_score = penalties.sum()
        mean_score = penalties.mean()

        overprediction = (errors > 0).sum()
        underprediction = (errors < 0).sum()
        exact_prediction = (errors == 0).sum()

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "PHM08_Total_Score": total_score,
            "PHM08_Mean_Score": mean_score,
            "Overprediction": int(overprediction),
            "Underprediction": int(underprediction),
            "Exact": int(exact_prediction),
        })

    # ==========================================
    # 3.3 DISPLAY COMPARISON
    # ==========================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="PHM08_Total_Score",
        ascending=True,
    )

    print("\n===== PHM08 MODEL COMPARISON =====")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==========================================
    # 3.4 SAVE REPORT
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR / "ml_models_phm08_score.csv"
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    print("\nPHM08 report saved:", report_path)


if __name__ == "__main__":
    main()
