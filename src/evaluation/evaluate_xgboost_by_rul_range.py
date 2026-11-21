
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
    / "xgboost_test_predictions.csv"
)

REPORT_DIR = BASE_DIR / "reports"


def main():

    # ==========================================
    # 2. LOAD TEST PREDICTIONS
    # ==========================================

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "unit_number",
        "actual_RUL",
        "predicted_RUL",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # ==========================================
    # 3. CALCULATE PREDICTION ERRORS
    # ==========================================

    df["error"] = (
        df["predicted_RUL"] - df["actual_RUL"]
    )

    df["absolute_error"] = df["error"].abs()

    # ==========================================
    # 4. DIVIDE ENGINES INTO RUL GROUPS
    # ==========================================

    bins = [-np.inf, 30, 80, np.inf]

    labels = [
        "Low RUL (0-30)",
        "Medium RUL (30-80)",
        "High RUL (>80)",
    ]

    df["rul_group"] = pd.cut(
        df["actual_RUL"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    print("===== ENGINE DISTRIBUTION =====")

    print(
        df["rul_group"]
        .value_counts()
        .reindex(labels, fill_value=0)
        .to_string()
    )

    # ==========================================
    # 5. EVALUATE EACH RUL GROUP
    # ==========================================

    results = []

    for group_name, group_df in df.groupby(
        "rul_group",
        observed=True,
        sort=True,
    ):

        y_true = group_df["actual_RUL"]
        y_pred = group_df["predicted_RUL"]

        errors = group_df["error"]

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

        mean_error = errors.mean()

        overprediction = (errors > 0).sum()
        underprediction = (errors < 0).sum()

        results.append({
            "RUL_group": str(group_name),
            "n_engines": len(group_df),
            "MAE": mae,
            "RMSE": rmse,
            "Mean_Error": mean_error,
            "Overprediction": int(overprediction),
            "Underprediction": int(underprediction),
        })

    # ==========================================
    # 6. DISPLAY RESULTS
    # ==========================================

    results_df = pd.DataFrame(results)

    print("\n===== XGBOOST PERFORMANCE BY RUL GROUP =====")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ==========================================
    # 7. SAVE REPORT
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR
        / "xgboost_performance_by_rul_group.csv"
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
