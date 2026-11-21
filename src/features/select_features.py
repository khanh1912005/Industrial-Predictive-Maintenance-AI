
from pathlib import Path
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_clean.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_REPORT = (
    REPORT_DIR / "feature_selection_report.csv"
)

CORRELATION_REPORT = (
    REPORT_DIR / "high_correlation_pairs.csv"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay: {DATA_FILE}\n"
        "Hay chay clean_data.py truoc."
    )

df = pd.read_csv(DATA_FILE)

print("\n===== FEATURE SELECTION =====")
print(f"Dataset shape: {df.shape}")


# ==========================================
# 3. DEFINE FEATURE GROUPS
# ==========================================

setting_columns = [
    f"setting_{i}" for i in range(1, 4)
]

sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

metadata_columns = [
    "unit_number",
    "time_cycles"
]

input_columns = setting_columns + sensor_columns

required_columns = metadata_columns + input_columns

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thieu cot du lieu: {missing_columns}"
    )


# ==========================================
# 4. IDENTIFY CONSTANT FEATURES
# ==========================================

print("\n===== CONSTANT FEATURES =====")

feature_summary = pd.DataFrame({
    "feature": input_columns,

    "feature_type": (
        ["operational_setting"] * len(setting_columns)
        + ["sensor"] * len(sensor_columns)
    ),

    "unique_values": [
        df[col].nunique(dropna=False)
        for col in input_columns
    ],

    "std": [
        df[col].std()
        for col in input_columns
    ]
})

feature_summary["is_constant"] = (
    feature_summary["unique_values"] <= 1
)

constant_features = feature_summary.loc[
    feature_summary["is_constant"],
    "feature"
].tolist()

candidate_features = feature_summary.loc[
    ~feature_summary["is_constant"],
    "feature"
].tolist()

print("Features khong bien dong:")
print(constant_features)

print(f"\nSo feature ban dau: {len(input_columns)}")
print(f"So feature hang: {len(candidate_features)}")
print(f"So feature bi loai bo: {len(constant_features)}")


# ==========================================
# 5. ANALYZE HIGH CORRELATION
# ==========================================

print("\n===== HIGH CORRELATION FEATURES =====")

CORRELATION_THRESHOLD = 0.80

correlation_matrix = df[candidate_features].corr(
    method="pearson"
)

correlation_pairs = []

for i in range(len(candidate_features)):

    for j in range(i + 1, len(candidate_features)):

        feature_a = candidate_features[i]
        feature_b = candidate_features[j]

        correlation = correlation_matrix.loc[
            feature_a, feature_b
        ]

        if abs(correlation) >= CORRELATION_THRESHOLD:

            correlation_pairs.append({
                "feature_a": feature_a,
                "feature_b": feature_b,
                "pearson_r": correlation,
                "absolute_correlation": abs(correlation)
            })


# Convert results to DataFrame

pairs_df = pd.DataFrame(
    correlation_pairs,
    columns=[
        "feature_a",
        "feature_b",
        "pearson_r",
        "absolute_correlation"
    ]
)

if not pairs_df.empty:

    pairs_df = pairs_df.sort_values(
        "absolute_correlation",
        ascending=False
    )

    print(pairs_df.round(4).to_string(index=False))

else:

    print("Khong co cap feature vuot nguong.")


# ==========================================
# 6. SAVE FEATURE REPORTS
# ==========================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

feature_summary.to_csv(
    FEATURE_REPORT,
    index=False
)

pairs_df.to_csv(
    CORRELATION_REPORT,
    index=False
)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n===== FEATURE SELECTION SUMMARY =====")

print(f"Total input features: {len(input_columns)}")
print(f"Constant features: {len(constant_features)}")
print(f"Candidate features: {len(candidate_features)}")

print("\nCandidate features:")
print(candidate_features)

print("\nReports saved successfully:")
print(FEATURE_REPORT)
print(CORRELATION_REPORT)

print("\nFEATURE ANALYSIS COMPLETED")
