
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
    # 2. LOAD PREDICTION DATA
    # ==========================================

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "actual_RUL",
        "predicted_RUL",
    ]

    if not all(column in df.columns for column in required_columns):
        raise ValueError("Missing required prediction columns!")

    # Recalculate error from official NASA RUL
    df["error"] = (
        df["predicted_RUL"] - df["actual_RUL"]
    )

    print("Number of engines:", len(df))

    # ==========================================
    # 3. PREDICTED VS ACTUAL RUL
    # ==========================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["actual_RUL"],
        df["predicted_RUL"],
        alpha=0.7,
    )

    min_value = min(
        df["actual_RUL"].min(),
        df["predicted_RUL"].min(),
    )

    max_value = max(
        df["actual_RUL"].max(),
        df["predicted_RUL"].max(),
    )

    margin = max((max_value - min_value) * 0.05, 1)

    min_value -= margin
    max_value += margin

    # Perfect prediction reference line
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Prediction",
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("XGBoost: Predicted vs Actual RUL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "xgboost_predicted_vs_actual.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ==========================================
    # 4. ERROR DISTRIBUTION
    # ==========================================

    plt.figure(figsize=(8, 6))

    plt.hist(
        df["error"],
        bins=15,
    )

    plt.axvline(
        x=0,
        linestyle="--",
        label="Zero Error",
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Number of Engines")
    plt.title("XGBoost: Error Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "xgboost_error_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ==========================================
    # 5. ERROR VS ACTUAL RUL
    # ==========================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["actual_RUL"],
        df["error"],
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        label="Zero Error",
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Prediction Error")
    plt.title("XGBoost: Error vs Actual RUL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "xgboost_error_vs_actual_rul.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("\nAll error visualization plots saved successfully!")


if __name__ == "__main__":
    main()
