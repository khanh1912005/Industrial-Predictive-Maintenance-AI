
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "scalers"
)

INPUT_TRAIN = (
    DATA_DIR / "train_FD001_model_train.csv"
)

INPUT_VAL = (
    DATA_DIR / "train_FD001_validation.csv"
)


# ==========================================
# 2. OUTPUT PATHS
# ==========================================

OUTPUT_FILES = {
    "standard_train": (
        DATA_DIR / "train_FD001_standard_scaled.csv"
    ),
    "standard_val": (
        DATA_DIR / "validation_FD001_standard_scaled.csv"
    ),
    "minmax_train": (
        DATA_DIR / "train_FD001_minmax_scaled.csv"
    ),
    "minmax_val": (
        DATA_DIR / "validation_FD001_minmax_scaled.csv"
    )
}


# ==========================================
# 3. LOAD DATA
# ==========================================

print("\n===== LOAD DATA =====")

if not INPUT_TRAIN.is_file():
    raise FileNotFoundError(
        f"Khong tim thay: {INPUT_TRAIN}"
    )

if not INPUT_VAL.is_file():
    raise FileNotFoundError(
        f"Khong tim thay: {INPUT_VAL}"
    )

train_df = pd.read_csv(INPUT_TRAIN)
val_df = pd.read_csv(INPUT_VAL)

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")


# ==========================================
# 4. VERIFY ENGINE SEPARATION
# ==========================================

train_engines = set(train_df["unit_number"])
val_engines = set(val_df["unit_number"])

overlap = train_engines.intersection(val_engines)

if overlap:
    raise ValueError(
        f"Train va Validation bi trung engine: {overlap}"
    )


# ==========================================
# 5. DEFINE INPUT FEATURES
# ==========================================

setting_columns = [
    f"setting_{i}" for i in range(1, 4)
]

sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

input_candidates = setting_columns + sensor_columns

for col in input_candidates:
    if col not in train_df.columns or col not in val_df.columns:
        raise ValueError(f"Thieu cot: {col}")


# ==========================================
# 6. REMOVE CONSTANT FEATURES
# ==========================================

print("\n===== CONSTANT FEATURE CHECK =====")

constant_features = [
    col for col in input_candidates
    if train_df[col].nunique(dropna=False) <= 1
]

feature_columns = [
    col for col in input_candidates
    if col not in constant_features
]

if not feature_columns:
    raise ValueError("Khong con feature hop le.")

print(f"Constant features: {constant_features}")
print(f"Total candidate features: {len(input_candidates)}")
print(f"Features used for scaling: {len(feature_columns)}")

# Chỉ sử dụng thông tin Training Set để
# quyết định đặc trưng nào có biến thiên.


# ==========================================
# 7. DEFINE CONTEXT AND TARGET COLUMNS
# ==========================================

context_columns = [
    "unit_number",
    "time_cycles",
    "RUL_raw",
    "RUL"
]

for col in context_columns:
    if col not in train_df.columns or col not in val_df.columns:
        raise ValueError(f"Thieu cot context/target: {col}")


# ==========================================
# 8. CREATE OUTPUT DATAFRAME
# ==========================================

def build_scaled_dataframe(
    original_df,
    scaled_values,
    features
):
    """
    Ghep metadata, target va feature da scale.
    """

    context = original_df[
        context_columns
    ].reset_index(drop=True)

    scaled_features = pd.DataFrame(
        scaled_values,
        columns=features
    )

    return pd.concat(
        [context, scaled_features],
        axis=1
    )


# ==========================================
# 9. STANDARD SCALER
# ==========================================

print("\n===== STANDARD SCALER =====")

standard_scaler = StandardScaler()

# FIT + TRANSFORM tren Training Set
X_train_standard = standard_scaler.fit_transform(
    train_df[feature_columns]
)

# CHI TRANSFORM tren Validation Set
X_val_standard = standard_scaler.transform(
    val_df[feature_columns]
)

standard_train_df = build_scaled_dataframe(
    train_df,
    X_train_standard,
    feature_columns
)

standard_val_df = build_scaled_dataframe(
    val_df,
    X_val_standard,
    feature_columns
)

print("Training mean after scaling:")
print(
    standard_train_df[feature_columns]
    .mean()
    .round(4)
    .to_string()
)


# ==========================================
# 10. MINMAX SCALER
# ==========================================

print("\n===== MINMAX SCALER =====")

minmax_scaler = MinMaxScaler()

# FIT + TRANSFORM tren Training Set
X_train_minmax = minmax_scaler.fit_transform(
    train_df[feature_columns]
)

# CHI TRANSFORM tren Validation Set
X_val_minmax = minmax_scaler.transform(
    val_df[feature_columns]
)

minmax_train_df = build_scaled_dataframe(
    train_df,
    X_train_minmax,
    feature_columns
)

minmax_val_df = build_scaled_dataframe(
    val_df,
    X_val_minmax,
    feature_columns
)

print("Training min after scaling:")
print(
    minmax_train_df[feature_columns]
    .min()
    .round(4)
    .to_string()
)

print("\nTraining max after scaling:")
print(
    minmax_train_df[feature_columns]
    .max()
    .round(4)
    .to_string()
)


# ==========================================
# 11. SAVE SCALED DATASETS
# ==========================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

standard_train_df.to_csv(
    OUTPUT_FILES["standard_train"],
    index=False
)

standard_val_df.to_csv(
    OUTPUT_FILES["standard_val"],
    index=False
)

minmax_train_df.to_csv(
    OUTPUT_FILES["minmax_train"],
    index=False
)

minmax_val_df.to_csv(
    OUTPUT_FILES["minmax_val"],
    index=False
)


# ==========================================
# 12. SAVE SCALERS
# ==========================================

joblib.dump(
    standard_scaler,
    ARTIFACT_DIR / "standard_scaler.joblib"
)

joblib.dump(
    minmax_scaler,
    ARTIFACT_DIR / "minmax_scaler.joblib"
)


# ==========================================
# 13. SAVE FEATURE METADATA
# ==========================================

metadata = {
    "input_candidates": input_candidates,
    "constant_features": constant_features,
    "selected_features": feature_columns,
    "context_columns": context_columns,
    "target_columns": ["RUL_raw", "RUL"]
}

with open(
    ARTIFACT_DIR / "feature_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n===== SCALING COMPLETED =====")

print(f"Selected features: {len(feature_columns)}")

for name, path in OUTPUT_FILES.items():
    print(f"{name}: {path}")

print("\nScalers saved successfully.")
