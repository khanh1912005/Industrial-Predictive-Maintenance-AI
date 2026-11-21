
from pathlib import Path
import pandas as pd


# ==========================================
# 1. XAC DINH DUONG DAN DU LIEU
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
    / "train_FD001.txt"
)


# ==========================================
# 2. KHAI BAO TEN 26 COT
# ==========================================

column_names = [
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
]

for i in range(1, 22):
    column_names.append(f"sensor_{i}")


# ==========================================
# 3. DOC DU LIEU
# ==========================================

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file: {DATA_FILE}"
    )

train_df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None,
    names=column_names
)


# ==========================================
# 4. KIEM TRA KICH THUOC
# ==========================================

print("\n===== DATASET INFORMATION =====")

print(f"So dong: {train_df.shape[0]}")
print(f"So cot: {train_df.shape[1]}")
print(
    f"So dong co: "
    f"{train_df['unit_number'].nunique()}"
)


# ==========================================
# 5. KIEM TRA MISSING VALUES
# ==========================================

print("\n===== MISSING VALUES =====")

missing_values = train_df.isna().sum()

total_missing = int(missing_values.sum())

print(f"Tong so gia tri bi thieu: {total_missing}")

if total_missing > 0:
    print(missing_values[missing_values > 0])
else:
    print("Khong co missing values.")


# ==========================================
# 6. KIEM TRA DUPLICATE ROWS
# ==========================================

print("\n===== DUPLICATE ROWS =====")

duplicate_rows = int(train_df.duplicated().sum())

print(f"So dong bi trung hoan toan: {duplicate_rows}")


# ==========================================
# 7. KIEM TRA ENGINE-CYCLE BI TRUNG
# ==========================================

print("\n===== DUPLICATE ENGINE-CYCLE =====")

duplicate_cycles = int(
    train_df.duplicated(
        subset=["unit_number", "time_cycles"]
    ).sum()
)

print(f"So cap Engine-Cycle bi trung: {duplicate_cycles}")


# ==========================================
# 8. KIEM TRA TINH LIEN TUC CUA CHU KY
# ==========================================

print("\n===== CYCLE CONTINUITY =====")

sorted_df = train_df.sort_values(
    ["unit_number", "time_cycles"]
)

cycle_diff = (
    sorted_df
    .groupby("unit_number")["time_cycles"]
    .diff()
)

invalid_cycle_steps = int(
    (cycle_diff.notna() & (cycle_diff != 1)).sum()
)

first_cycles = (
    sorted_df.groupby("unit_number")["time_cycles"].min()
)

invalid_first_cycles = int(
    (first_cycles != 1).sum()
)

print(f"So buoc chu ky khong lien tuc: {invalid_cycle_steps}")
print(f"So dong co khong bat dau tu chu ky 1: {invalid_first_cycles}")


# ==========================================
# 9. KIEM TRA DATA TYPES
# ==========================================

print("\n===== DATA TYPES =====")

non_numeric_columns = train_df.select_dtypes(
    exclude="number"
).columns.tolist()

if non_numeric_columns:
    print(f"Cot khong phai kieu so: {non_numeric_columns}")
else:
    print("Tat ca cac cot deu la du lieu so.")


# ==========================================
# 10. TONG KET
# ==========================================

print("\n===== DATA QUALITY SUMMARY =====")

quality_passed = (
    total_missing == 0
    and duplicate_rows == 0
    and duplicate_cycles == 0
    and invalid_cycle_steps == 0
    and invalid_first_cycles == 0
    and len(non_numeric_columns) == 0
)

if not quality_passed:

    print("DATA QUALITY CHECK: NEEDS REVIEW")

    raise ValueError(
        "Dataset failed quality validation. "
        "Pipeline must stop."
    )

print("DATA QUALITY CHECK: PASSED")