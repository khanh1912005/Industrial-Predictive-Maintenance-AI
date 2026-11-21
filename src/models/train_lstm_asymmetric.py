
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

WINDOW_DIR = BASE_DIR / "data" / "processed" / "windows"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "lstm_asymmetric_alpha3_beta1.pt"
HISTORY_PATH = REPORT_DIR / "lstm_asymmetric_training_history.csv"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64

ALPHA = 3.0
BETA = 1.0

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
# 2. ASYMMETRIC LOSS FUNCTION
# ==========================================

class AsymmetricMSELoss(nn.Module):

    def __init__(self, alpha=3.0, beta=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):

        errors = predictions - targets

        overprediction_loss = (
            self.alpha * errors.pow(2)
        )

        underprediction_loss = (
            self.beta * errors.pow(2)
        )

        loss = torch.where(
            errors > 0,
            overprediction_loss,
            underprediction_loss
        )

        return loss.mean()


# ==========================================
# 3. PHM08 METRIC FUNCTION
# ==========================================

def calculate_phm08(errors):

    scores = np.zeros_like(
        errors,
        dtype=np.float64
    )

    over_mask = errors >= 0
    under_mask = errors < 0

    scores[over_mask] = np.expm1(
        errors[over_mask] / 10
    )

    scores[under_mask] = np.expm1(
        -errors[under_mask] / 13
    )

    return float(scores.sum())


# ==========================================
# 4. LOAD DATA
# ==========================================

X_train = np.load(WINDOW_DIR / "X_train.npy")
y_train = np.load(WINDOW_DIR / "y_train.npy")

X_validation = np.load(
    WINDOW_DIR / "X_validation.npy"
)

y_validation = np.load(
    WINDOW_DIR / "y_validation.npy"
)

print("\n===== DATA SHAPES =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)


# ==========================================
# 5. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_validation = torch.from_numpy(
    X_validation
).float()

y_validation = torch.from_numpy(
    y_validation
).float()


# ==========================================
# 6. CREATE DATALOADERS
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
# 7. INITIALIZE MODEL AND LOSS
# ==========================================

input_size = X_train.shape[2]

model = LSTMRegressor(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

criterion = AsymmetricMSELoss(
    alpha=ALPHA,
    beta=BETA
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n===== MODEL CONFIGURATION =====")

print("Input size:", input_size)
print("Hidden size:", HIDDEN_SIZE)
print("Alpha:", ALPHA)
print("Beta:", BETA)

print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)


# ==========================================
# 8. TRAINING LOOP
# ==========================================

best_validation_mae = float("inf")
best_epoch = 0

best_validation_rmse = None
best_validation_phm08 = None

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

    train_asymmetric_loss = (
        total_train_loss / train_count
    )


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_x, batch_y in validation_loader:

            batch_x = batch_x.to(DEVICE)

            predictions = model(batch_x)

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                batch_y.numpy()
            )

    all_predictions = np.concatenate(
        all_predictions
    )

    all_targets = np.concatenate(
        all_targets
    )

    errors = all_predictions - all_targets

    validation_mae = np.mean(
        np.abs(errors)
    )

    validation_rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    validation_phm08 = calculate_phm08(
        errors
    )

    overprediction_count = int(
        np.sum(errors > 0)
    )

    underprediction_count = int(
        np.sum(errors < 0)
    )


    # --------------------------------------
    # SAVE HISTORY
    # --------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_asymmetric_loss": train_asymmetric_loss,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_phm08": validation_phm08,
        "overprediction": overprediction_count,
        "underprediction": underprediction_count
    })


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae
        best_epoch = epoch + 1

        best_validation_rmse = validation_rmse
        best_validation_phm08 = validation_phm08

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,

                "alpha": ALPHA,
                "beta": BETA,

                "best_epoch": best_epoch,

                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
                "validation_phm08": validation_phm08
            },
            MODEL_PATH
        )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Asym Loss: {train_asymmetric_loss:.4f} | "
        f"Val MAE: {validation_mae:.4f} | "
        f"Val RMSE: {validation_rmse:.4f} | "
        f"Val PHM08: {validation_phm08:.4f}"
    )


# ==========================================
# 9. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ==========================================
# 10. FINAL RESULTS
# ==========================================

print("\n===== ASYMMETRIC LSTM TRAINING COMPLETED =====")

print("Alpha:", ALPHA)
print("Beta:", BETA)

print("Best epoch:", best_epoch)

print("Best validation MAE:", best_validation_mae)
print("Validation RMSE:", best_validation_rmse)
print("Validation PHM08:", best_validation_phm08)

print("Model saved:", MODEL_PATH)
print("History saved:", HISTORY_PATH)
