
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from src.models.cnn_model import CNN1DRegressor


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

WINDOW_DIR = BASE_DIR / "data" / "processed" / "windows"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "cnn.pt"
HISTORY_PATH = REPORT_DIR / "cnn_training_history.csv"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)


# ==========================================
# 2. LOAD DATA
# ==========================================

X_train = np.load(
    WINDOW_DIR / "X_train.npy"
).astype(np.float32)

y_train = np.load(
    WINDOW_DIR / "y_train.npy"
).astype(np.float32)

X_validation = np.load(
    WINDOW_DIR / "X_validation.npy"
).astype(np.float32)

y_validation = np.load(
    WINDOW_DIR / "y_validation.npy"
).astype(np.float32)

print("\n===== DATA SHAPES =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)


# ==========================================
# 3. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_validation = torch.from_numpy(X_validation)
y_validation = torch.from_numpy(y_validation)


# ==========================================
# 4. CREATE DATALOADERS
# ==========================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

validation_dataset = TensorDataset(
    X_validation,
    y_validation
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. INITIALIZE MODEL
# ==========================================

model = CNN1DRegressor(
    input_channels=X_train.shape[2]
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n===== MODEL CONFIGURATION =====")

print("Input channels:", X_train.shape[2])
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)


# ==========================================
# 6. TRAINING LOOP
# ==========================================

best_validation_mae = float("inf")
best_epoch = 0

history = []

for epoch in range(EPOCHS):

    # --------------------------------------
    # TRAINING
    # --------------------------------------

    model.train()

    total_train_loss = 0.0
    train_count = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_x)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * batch_y.size(0)
        )

        train_count += batch_y.size(0)

    train_mse = total_train_loss / train_count


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    validation_squared_error = 0.0
    validation_absolute_error = 0.0
    validation_count = 0

    with torch.no_grad():

        for batch_x, batch_y in validation_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)

            errors = predictions - batch_y

            validation_squared_error += (
                torch.square(errors).sum().item()
            )

            validation_absolute_error += (
                torch.abs(errors).sum().item()
            )

            validation_count += batch_y.numel()

    validation_mse = (
        validation_squared_error / validation_count
    )

    validation_mae = (
        validation_absolute_error / validation_count
    )

    validation_rmse = np.sqrt(validation_mse)


    # --------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae
    })


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae
        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "input_channels": int(X_train.shape[2]),

                "best_epoch": int(best_epoch),

                "validation_mae": float(
                    validation_mae
                ),

                "validation_rmse": float(
                    validation_rmse
                )
            },
            MODEL_PATH
        )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train MSE: {train_mse:.4f} | "
        f"Val MSE: {validation_mse:.4f} | "
        f"Val RMSE: {validation_rmse:.4f} | "
        f"Val MAE: {validation_mae:.4f}"
    )


# ==========================================
# 7. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ==========================================
# 8. FINAL RESULTS
# ==========================================

print("\n===== CNN TRAINING COMPLETED =====")

print("Best epoch:", best_epoch)

print(
    "Best validation MAE:",
    best_validation_mae
)

print("Best model saved:", MODEL_PATH)
print("Training history saved:", HISTORY_PATH)
