
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "reports"

INPUT_PATH = REPORT_DIR / "lstm_test_predictions.csv"

GROUP_REPORT_PATH = REPORT_DIR / "lstm_test_error_by_rul.csv"
TOP_ERRORS_PATH = REPORT_DIR / "lstm_test_top_errors.csv"


# ==========================================
# 2. LOAD PREDICTIONS
# ==========================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "unit_number",
    "actual_RUL",
    "predicted_RUL",
    "error",
    "absolute_error"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("Total test engines:", len(df))


# ==========================================
# 3. CREATE RUL GROUPS
# ==========================================

df["RUL_range"] = pd.cut(
    df["actual_RUL"],
    bins=[-np.inf, 30, 80, np.inf],
    labels=[
        "Low (<=30)",
        "Medium (30-80]",
        "High (>80)"
    ]
)


# ==========================================
# 4. ANALYZE EACH RUL GROUP
# ==========================================

group_results = []

for group_name, group in df.groupby(
    "RUL_range",
    observed=True
):

    errors = group["error"].to_numpy()

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mean_error = np.mean(errors)

    overprediction = int(
        np.sum(errors > 0)
    )

    underprediction = int(
        np.sum(errors < 0)
    )

    phm08 = np.where(
        errors < 0,
        np.expm1(-errors / 13),
        np.expm1(errors / 10)
    ).sum()

    group_results.append({
        "RUL_range": group_name,
        "Engine_count": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "Mean_Error": mean_error,
        "Overprediction": overprediction,
        "Underprediction": underprediction,
        "PHM08_Score": phm08
    })


group_report = pd.DataFrame(group_results)


# ==========================================
# 5. FIND TOP 10 WORST PREDICTIONS
# ==========================================

top_errors = (
    df.sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
    .copy()
)


# ==========================================
# 6. SAVE REPORTS
# ==========================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

group_report.to_csv(
    GROUP_REPORT_PATH,
    index=False
)

top_errors.to_csv(
    TOP_ERRORS_PATH,
    index=False
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n===== LSTM ERROR BY RUL RANGE =====")

print(
    group_report.round(4).to_string(index=False)
)

print("\n===== TOP 10 WORST LSTM PREDICTIONS =====")

print(
    top_errors[
        [
            "unit_number",
            "actual_RUL",
            "predicted_RUL",
            "error",
            "absolute_error"
        ]
    ].round(4).to_string(index=False)
)


print("\n===== SAVED REPORTS =====")

print("RUL group report:", GROUP_REPORT_PATH)
print("Top errors report:", TOP_ERRORS_PATH)
