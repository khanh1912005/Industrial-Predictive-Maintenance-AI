
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_rul_capped.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_OUTPUT = (
    OUTPUT_DIR / "train_FD001_model_train.csv"
)

VAL_OUTPUT = (
    OUTPUT_DIR / "train_FD001_validation.csv"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\n===== LOAD DATA =====")

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file: {INPUT_FILE}\n"
        "Hay chay build_rul_labels.py truoc."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df)}")
print(f"Total engines: {df['unit_number'].nunique()}")


# ==========================================
# 3. GROUP SPLITTING
# ==========================================

print("\n===== GROUP TRAIN VALIDATION SPLIT =====")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, val_indices = next(
    splitter.split(
        df,
        groups=df["unit_number"]
    )
)


# ==========================================
# 4. CREATE DATASETS
# ==========================================

train_df = df.iloc[train_indices].copy()

val_df = df.iloc[val_indices].copy()


# ==========================================
# 5. SORT DATA
# ==========================================

train_df = train_df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)

val_df = val_df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)


# ==========================================
# 6. VERIFY NO ENGINE OVERLAP
# ==========================================

train_engines = set(train_df["unit_number"])

val_engines = set(val_df["unit_number"])

overlap = train_engines.intersection(val_engines)

if overlap:
    raise ValueError(
        f"Engine bi trung giua Train va Validation: {overlap}"
    )


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n===== SPLIT RESULTS =====")

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")

print(f"Training engines: {len(train_engines)}")
print(f"Validation engines: {len(val_engines)}")

print(f"Engine overlap: {len(overlap)}")


# ==========================================
# 8. SAVE DATASETS
# ==========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)

val_df.to_csv(
    VAL_OUTPUT,
    index=False
)

print("\n===== SAVE COMPLETED =====")

print(f"Training: {TRAIN_OUTPUT}")
print(f"Validation: {VAL_OUTPUT}")

print("\nTRAIN VALIDATION SPLIT COMPLETED")
