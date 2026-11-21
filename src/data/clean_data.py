
from pathlib import Path
import pandas as pd


# ==========================================
# 1. XAC DINH DUONG DAN
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
    / "train_FD001.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_FD001_clean.csv"
)


# ==========================================
# 2. KHAI BAO TEN COT
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

print("\n===== LOADING DATA =====")

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file: {DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None,
    names=column_names
)

print(f"So dong ban dau: {len(df)}")


# ==========================================
# 4. CHUAN HOA KIEU DU LIEU SO
# ==========================================

print("\n===== CHECK DATA TYPES =====")

for col in column_names:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Neu co gia tri khong the chuyen thanh so,
# chung se tro thanh NaN.


# ==========================================
# 5. KIEM TRA MISSING VALUES
# ==========================================

print("\n===== CHECK MISSING VALUES =====")

missing_values = df.isna().sum()

total_missing = int(missing_values.sum())

print(f"Tong missing values: {total_missing}")

if total_missing > 0:
    print("Cac cot bi thieu du lieu:")
    print(missing_values[missing_values > 0])

    raise ValueError(
        "Phat hien missing values. "
        "Can chon chinh sach xu ly truoc khi tiep tuc."
    )


# ==========================================
# 6. XU LY DUPLICATE ROWS
# ==========================================

print("\n===== REMOVE EXACT DUPLICATES =====")

duplicate_rows = int(df.duplicated().sum())

print(f"So dong trung hoan toan: {duplicate_rows}")

if duplicate_rows > 0:
    df = df.drop_duplicates().copy()

print(f"So dong sau xu ly: {len(df)}")


# ==========================================
# 7. KIEM TRA ENGINE-CYCLE
# ==========================================

print("\n===== CHECK ENGINE-CYCLE =====")

duplicate_cycles = int(
    df.duplicated(
        subset=["unit_number", "time_cycles"]
    ).sum()
)

print(f"So cap Engine-Cycle trung: {duplicate_cycles}")

if duplicate_cycles > 0:
    raise ValueError(
        "Phat hien mot Engine-Cycle co nhieu ban ghi. "
        "Can dieu tra du lieu nguon."
    )


# ==========================================
# 8. KIEM TRA TINH LIEN TUC
# ==========================================

print("\n===== CHECK CYCLE CONTINUITY =====")

df = df.sort_values(
    ["unit_number", "time_cycles"]
).reset_index(drop=True)

cycle_diff = (
    df.groupby("unit_number")["time_cycles"]
    .diff()
)

invalid_steps = (
    cycle_diff.notna() & (cycle_diff != 1)
)

if invalid_steps.any():
    raise ValueError(
        "Phat hien chu ky khong lien tuc. "
        "Can kiem tra du lieu truoc khi xu ly."
    )

print("Cycle continuity: PASSED")


# ==========================================
# 9. LUU CLEAN DATASET
# ==========================================

print("\n===== SAVE CLEAN DATA =====")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"File da duoc luu tai: {OUTPUT_FILE}")

print(f"So dong cuoi cung: {len(df)}")
print(f"So cot cuoi cung: {len(df.columns)}")

print("\nDATA CLEANING COMPLETED")
