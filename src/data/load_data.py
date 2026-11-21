
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 2. Xác định vị trí file dữ liệu NASA
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
    / "train_FD001.txt"
)


# 3. Đặt tên cho 26 cột dữ liệu
column_names = [
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
]

# Tạo tên cho 21 cảm biến
for sensor_number in range(1, 22):
    column_names.append(f"sensor_{sensor_number}")


# 4. Kiểm tra file dữ liệu có tồn tại không
if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file du lieu: {DATA_FILE}"
    )


# 5. Đọc dữ liệu bằng Pandas
train_df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None,
    names=column_names
)


# 6. Hiển thị 5 dòng đầu tiên
print("\n===== 5 DONG DAU TIEN =====")
print(train_df.head())


# 7. Hiển thị kích thước dữ liệu
print("\n===== KICH THUOC DATAFRAME =====")
print(f"So dong: {train_df.shape[0]}")
print(f"So cot: {train_df.shape[1]}")


# 8. Hiển thị tên các cột
print("\n===== TEN CAC COT =====")
print(train_df.columns.tolist())

# 9. Đếm số lượng động cơ
print("\n===== SO LUONG DONG CO =====")

number_of_engines = train_df["unit_number"].nunique()

print(f"So luong dong co: {number_of_engines}")

# 10. Kiểm tra số chu kỳ của từng động cơ
print("\n===== SO CHU KY CUA CAC DONG CO =====")

cycles_per_engine = (
    train_df.groupby("unit_number")["time_cycles"].max()
)

print(cycles_per_engine.head())


# 11. Phân tích thống kê 21 cảm biến
print("\n===== THONG KE 21 CAM BIEN =====")

sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

sensor_stats = train_df[sensor_columns].describe().T

print(sensor_stats.round(3).to_string())


# 12. Xếp hạng cảm biến theo độ biến động
print("\n===== DO BIEN DONG CUA CAC CAM BIEN =====")

sensor_std = train_df[sensor_columns].std()

print(sensor_std.sort_values().round(5).to_string())

# 13. Chọn dữ liệu của động cơ số 1

engine_id = 1

engine_df = train_df[
    train_df["unit_number"] == engine_id
].copy()

print("\n===== THONG TIN DONG CO =====")
print(f"Engine ID: {engine_id}")
print(f"So chu ky: {len(engine_df)}")


# 14. Ve bieu do 3 cam bien cua dong co so 1

sensors_to_plot = [
    "sensor_2",
    "sensor_3",
    "sensor_4"
]

# Tao 3 bieu do rieng biet
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(12, 10),
    sharex=True
)

# Ve tung cam bien
for ax, sensor_name in zip(axes, sensors_to_plot):

    ax.plot(
        engine_df["time_cycles"],
        engine_df[sensor_name],
        linewidth=1.5
    )

    ax.set_title(f"Engine {engine_id} - {sensor_name}")
    ax.set_ylabel("Sensor value")
    ax.grid(True, alpha=0.3)

# Dat ten truc X
axes[-1].set_xlabel("Operating cycle")

fig.suptitle(
    f"Sensor Trends - Engine {engine_id}",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

# Luu bieu do
output_path = (
    PROJECT_ROOT
    / "reports"
    / "engine_1_sensor_trends.png"
)

fig.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nBieu do da duoc luu tai: {output_path}")

# Hien thi bieu do
plt.show()

# 15. Phan tich muc bien dong cua 21 cam bien
print("\n===== PHAN TICH BIEN DONG CAM BIEN =====")

sensor_analysis = train_df[sensor_columns].agg(
    ["min", "max", "std", "nunique"]
).T

# Tinh khoang bien thien
sensor_analysis["range"] = (
    sensor_analysis["max"] - sensor_analysis["min"]
)

# Sap xep theo do lech chuan tang dan
sensor_analysis = sensor_analysis.sort_values("std")

print(sensor_analysis.round(4).to_string())


# Tim cac cam bien khong thay doi
constant_sensors = sensor_analysis[
    sensor_analysis["nunique"] == 1
].index.tolist()

print("\n===== CAM BIEN KHONG THAY DOI =====")
print(constant_sensors)


# 16. Tao nhan RUL cho du lieu train
print("\n===== TAO NHAN RUL =====")

# Tim chu ky cuoi cung cua tung dong co
max_cycles = (
    train_df.groupby("unit_number")["time_cycles"].transform("max")
)

# Tinh RUL cho tung dong du lieu
train_df["RUL"] = max_cycles - train_df["time_cycles"]

# Kiem tra 5 dong dau tien
print("\n===== 5 DONG DAU TIEN =====")
print(
    train_df[
        ["unit_number", "time_cycles", "RUL"]
    ].head()
)

# Kiem tra 5 chu ky cuoi cua dong co so 1
print("\n===== 5 CHU KY CUOI ENGINE 1 =====")

engine_1_rul = train_df[
    train_df["unit_number"] == 1
]

print(
    engine_1_rul[
        ["unit_number", "time_cycles", "RUL"]
    ].tail()
)

# 17. Phan tich tuong quan giua sensor va RUL

print("\n===== TUONG QUAN SENSOR VA RUL =====")

# Loai bo cac sensor khong bien dong
active_sensors = [
    col for col in sensor_columns
    if train_df[col].nunique() > 1
]

# Tinh tuong quan Pearson
sensor_rul_corr = train_df[active_sensors].corrwith(
    train_df["RUL"]
)

# Sap xep tuong quan am den duong
sensor_rul_corr = sensor_rul_corr.sort_values()

print(sensor_rul_corr.round(4).to_string())


# 18. Ve Correlation Heatmap

print("\n===== CORRELATION HEATMAP =====")

# Chon 15 sensor co bien dong va cot RUL
correlation_columns = active_sensors + ["RUL"]

# Tinh ma tran tuong quan Pearson
correlation_matrix = train_df[
    correlation_columns
].corr(method="pearson")

# Tao bieu do
plt.figure(figsize=(15, 13))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 7}
)

plt.title("Correlation Heatmap - FD001 Train")
plt.tight_layout()

# Luu bieu do
heatmap_path = (
    PROJECT_ROOT
    / "reports"
    / "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=150,
    bbox_inches="tight"
)

print(f"Heatmap saved to: {heatmap_path}")

# Hien thi
plt.show()
