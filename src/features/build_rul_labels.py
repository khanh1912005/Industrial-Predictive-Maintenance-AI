 
from pathlib import Path
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_rul_capped.csv"
)


# ==========================================
# 2. RUL CONFIGURATION
# ==========================================

RUL_CAP = 125


# ==========================================
# 3. LOAD CLEAN DATA
# ==========================================

print("\n===== LOAD CLEAN DATA =====")

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file: {INPUT_FILE}\n"
        "Hay chay clean_data.py truoc."
    )

df = pd.read_csv(INPUT_FILE)

print(f"So dong: {len(df)}")


# ==========================================
# 4. VALIDATE REQUIRED COLUMNS
# ==========================================

required_columns = [
    "unit_number",
    "time_cycles"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Thieu cot bat buoc: {col}"
        )


# ==========================================
# 5. SORT DATA
# ==========================================

df = df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)


# ==========================================
# 6. CALCULATE RAW RUL
# ==========================================

print("\n===== CALCULATE RAW RUL =====")

max_cycles = (
    df.groupby("unit_number")["time_cycles"]
    .transform("max")
)

df["RUL_raw"] = (
    max_cycles - df["time_cycles"]
)

if (df["RUL_raw"] < 0).any():
    raise ValueError(
        "Phat hien RUL am. Hay kiem tra du lieu."
    )


# ==========================================
# 7. APPLY RUL CAPPING
# ==========================================

print("\n===== APPLY RUL CAPPING =====")

df["RUL"] = (
    df["RUL_raw"]
    .clip(upper=RUL_CAP)
    .astype("int64")
)

print(f"RUL cap: {RUL_CAP}")


# ==========================================
# 8. COMPARE RAW RUL AND CAPPED RUL
# ==========================================

print("\n===== RUL COMPARISON =====")

engine_id = 1

selected_cycles = [
    1, 2, 10, 50, 67, 100, 150, 192
]

comparison = df[
    (df["unit_number"] == engine_id)
    & (df["time_cycles"].isin(selected_cycles))
]

print(
    comparison[
        [
            "unit_number",
            "time_cycles",
            "RUL_raw",
            "RUL"
        ]
    ].to_string(index=False)
)


# ==========================================
# 9. RUL STATISTICS
# ==========================================

print("\n===== RUL STATISTICS =====")

print(f"Raw RUL max: {df['RUL_raw'].max()}")
print(f"Capped RUL max: {df['RUL'].max()}")

capped_rows = int(
    (df["RUL_raw"] > RUL_CAP).sum()
)

print(f"So dong bi cap: {capped_rows}")
print(
    f"So dong duoc giu nguyen: "
    f"{len(df) - capped_rows}"
)


# ==========================================
# 10. SAVE DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===== SAVE DATASET =====")
print(f"Saved to: {OUTPUT_FILE}")

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nRUL ENGINEERING COMPLETED")
