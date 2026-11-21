from pathlib import Path

import numpy as np
import pandas as pd


# 1. Xác định thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parents[2]

WINDOW_DIR = BASE_DIR / "data" / "processed" / "windows"


# 2. Load dữ liệu
X_train = np.load(WINDOW_DIR / "X_train.npy")
y_train = np.load(WINDOW_DIR / "y_train.npy")

X_validation = np.load(WINDOW_DIR / "X_validation.npy")
y_validation = np.load(WINDOW_DIR / "y_validation.npy")


# 3. Load thông tin window index
train_index = pd.read_csv(
    WINDOW_DIR / "train_window_index.csv"
)

validation_index = pd.read_csv(
    WINDOW_DIR / "validation_window_index.csv"
)


# 4. Kiểm tra shape
print("===== DATA SHAPES =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)


# 5. Kiểm tra kiểu dữ liệu
print("\n===== DATA TYPES =====")

print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)


# 6. Kiểm tra dữ liệu có NaN hoặc Inf
print("\n===== FINITE VALUE CHECK =====")

print("X_train finite:", np.isfinite(X_train).all())
print("y_train finite:", np.isfinite(y_train).all())

print(
    "X_validation finite:",
    np.isfinite(X_validation).all()
)

print(
    "y_validation finite:",
    np.isfinite(y_validation).all()
)


# 7. Kiểm tra sự đồng bộ giữa X, y và index
print("\n===== ALIGNMENT CHECK =====")

print(
    "Train X-Y match:",
    len(X_train) == len(y_train)
)

print(
    "Validation X-Y match:",
    len(X_validation) == len(y_validation)
)

print(
    "Train X-Index match:",
    len(X_train) == len(train_index)
)

print(
    "Validation X-Index match:",
    len(X_validation) == len(validation_index)
)


# 8. Kiểm tra phạm vi RUL
print("\n===== RUL LABEL RANGE =====")

print("Train RUL:", y_train.min(), "->", y_train.max())

print(
    "Validation RUL:",
    y_validation.min(),
    "->",
    y_validation.max()
)


# 9. Xem thông tin window index
print("\n===== WINDOW INDEX =====")

print("Train index columns:", train_index.columns.tolist())
print("Validation index columns:", validation_index.columns.tolist())

print("\nTrain index sample:")
print(train_index.head())