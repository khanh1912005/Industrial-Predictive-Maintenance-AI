
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_DIR = DATA_DIR / "windows"

METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

WINDOW_SIZE = 30
STRIDE = 1


# ==========================================
# 2. INPUT FILES
# ==========================================

INPUT_FILES = {
    "train": DATA_DIR / "train_FD001_standard_scaled.csv",
    "validation": DATA_DIR / "validation_FD001_standard_scaled.csv"
}


# ==========================================
# 3. LOAD FEATURE METADATA
# ==========================================

if not METADATA_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay metadata: {METADATA_FILE}"
    )

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

FEATURE_COLUMNS = metadata["selected_features"]

print("\n===== FEATURE INFORMATION =====")
print(f"Number of features: {len(FEATURE_COLUMNS)}")
print(f"Features: {FEATURE_COLUMNS}")


# ==========================================
# 4. CREATE SLIDING WINDOWS
# ==========================================

def create_windows(
    df,
    feature_columns,
    window_size,
    stride
):
    """
    Create time-series sliding windows separately
    for each engine.

    X shape:
        (number_of_windows, window_size, features)

    y shape:
        (number_of_windows,)
    """

    X_list = []
    y_list = []
    records = []

    skipped_engines = 0

    # Group records by engine
    for engine_id, group in df.groupby(
        "unit_number",
        sort=True
    ):

        # Sort each engine by operating cycle
        group = group.sort_values(
            "time_cycles"
        ).reset_index(drop=True)

        cycles = group["time_cycles"].to_numpy()

        # Check time-series continuity
        if not np.all(np.diff(cycles) == 1):
            raise ValueError(
                f"Engine {engine_id} co chu ky khong lien tuc."
            )

        # Skip engines shorter than window size
        if len(group) < window_size:
            skipped_engines += 1
            continue

        # Extract feature values and target
        feature_values = group[
            feature_columns
        ].to_numpy(dtype=np.float32)

        rul_values = group[
            "RUL"
        ].to_numpy(dtype=np.float32)

        # Generate sliding windows
        for end_idx in range(
            window_size - 1,
            len(group),
            stride
        ):

            start_idx = end_idx - window_size + 1

            window = feature_values[
                start_idx:end_idx + 1
            ]

            target = rul_values[end_idx]

            X_list.append(window)
            y_list.append(target)

            # Save information for each window
            records.append({
                "unit_number": int(engine_id),
                "start_cycle": int(cycles[start_idx]),
                "end_cycle": int(cycles[end_idx]),
                "RUL": float(target)
            })

    if not X_list:
        raise ValueError(
            "Khong tao duoc window nao."
        )

    X = np.stack(X_list).astype(np.float32)

    y = np.asarray(
        y_list,
        dtype=np.float32
    )

    window_index = pd.DataFrame(records)

    return X, y, window_index, skipped_engines


# ==========================================
# 5. LOAD DATASET
# ==========================================

def load_scaled_data(file_path):

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}\n"
            "Hay chay scale_features.py truoc."
        )

    return pd.read_csv(file_path)


# ==========================================
# 6. PROCESS TRAIN DATA
# ==========================================

print("\n===== CREATE TRAIN WINDOWS =====")

train_df = load_scaled_data(
    INPUT_FILES["train"]
)

X_train, y_train, train_index, skipped_train = (
    create_windows(
        train_df,
        FEATURE_COLUMNS,
        WINDOW_SIZE,
        STRIDE
    )
)


# ==========================================
# 7. PROCESS VALIDATION DATA
# ==========================================

print("\n===== CREATE VALIDATION WINDOWS =====")

val_df = load_scaled_data(
    INPUT_FILES["validation"]
)

X_val, y_val, val_index, skipped_val = (
    create_windows(
        val_df,
        FEATURE_COLUMNS,
        WINDOW_SIZE,
        STRIDE
    )
)


# ==========================================
# 8. VERIFY WINDOW DATA
# ==========================================

print("\n===== VERIFY WINDOW DATA =====")

for dataset_name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val)
]:

    if X.ndim != 3:
        raise ValueError(
            f"{dataset_name}: X phai co 3 dimensions."
        )

    if X.shape[1] != WINDOW_SIZE:
        raise ValueError(
            f"{dataset_name}: Sai window size."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{dataset_name}: X co NaN hoac infinity."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{dataset_name}: y co NaN hoac infinity."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{dataset_name}: So window va label khong khop."
        )

    print(
        f"{dataset_name} X shape: {X.shape}"
    )

    print(
        f"{dataset_name} y shape: {y.shape}"
    )


# ==========================================
# 9. VERIFY ENGINE SEPARATION
# ==========================================

train_engines = set(train_index["unit_number"])

val_engines = set(val_index["unit_number"])

overlap = train_engines.intersection(val_engines)

if overlap:
    raise ValueError(
        f"Engine overlap detected: {overlap}"
    )

print(f"Engine overlap: {len(overlap)}")


# ==========================================
# 10. SAVE WINDOW DATA
# ==========================================

WINDOW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    WINDOW_DIR / "X_train.npy",
    X_train
)

np.save(
    WINDOW_DIR / "y_train.npy",
    y_train
)

np.save(
    WINDOW_DIR / "X_validation.npy",
    X_val
)

np.save(
    WINDOW_DIR / "y_validation.npy",
    y_val
)

train_index.to_csv(
    WINDOW_DIR / "train_window_index.csv",
    index=False
)

val_index.to_csv(
    WINDOW_DIR / "validation_window_index.csv",
    index=False
)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n===== WINDOWING SUMMARY =====")

print(f"Window size: {WINDOW_SIZE}")
print(f"Stride: {STRIDE}")
print(f"Features: {len(FEATURE_COLUMNS)}")

print(f"Train windows: {len(X_train)}")
print(f"Validation windows: {len(X_val)}")

print(f"Skipped train engines: {skipped_train}")
print(f"Skipped validation engines: {skipped_val}")

print(f"\nOutput directory: {WINDOW_DIR}")

print("\nTIME-SERIES WINDOWING COMPLETED")
