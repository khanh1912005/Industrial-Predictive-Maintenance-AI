
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

MODEL_PATH = MODEL_DIR / "lstm_rul_aware.pt"
HISTORY_PATH = REPORT_DIR / "lstm_rul_aware_history.csv"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64

# RUL-aware loss weights
ALPHA_LOW = 3.0
ALPHA_MEDIUM = 1.5
ALPHA_HIGH = 1.0

BETA = 1.0

LOW_RUL_THRESHOLD = 30
MEDIUM_RUL_THRESHOLD = 80

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
# 2. RUL-AWARE ASYMMETRIC LOSS
# ==========================================

class RULAwareAsymmetricLoss(nn.Module):

    def __init__(
        self,
        alpha_low=3.0,
        alpha_medium=1.5,
        alpha_high=1.0,
        beta=1.0
    ):
        super().__init__()

        self.alpha_low = alpha_low
        self.alpha_medium = alpha_medium
        self.alpha_high = alpha_high
        self.beta = beta

    def forward(self, predictions, targets):

        errors = predictions - targets

        # Select overprediction weight
        # based on the actual RUL group

        alpha = torch.where(
            targets <= LOW_RUL_THRESHOLD,
            self.alpha_low,
            torch.where(
                targets <= MEDIUM_RUL_THRESHOLD,
                self.alpha_medium,
                self.alpha_high
            )
        )

        # Overprediction receives RUL-aware weight
        # Underprediction receives beta

        overprediction_loss = (
            alpha * errors.pow(2)
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
# 3. EVALUATION FUNCTIONS
# ==========================================

def calculate_phm08(errors):

    scores = np.where(
        errors < 0,
        np.expm1(-errors / 13),
        np.expm1(errors / 10)
    )

    return float(scores.sum())


def calculate_group_metrics(errors, targets):

    groups = {
        "low": targets <= LOW_RUL_THRESHOLD,

        "medium": (
            (targets > LOW_RUL_THRESHOLD)
            & (targets <= MEDIUM_RUL_THRESHOLD)
        ),

        "high": targets > MEDIUM_RUL_THRESHOLD
    }

    results = {}

    for group_name, mask in groups.items():

        group_errors = errors[mask]

        if len(group_errors) == 0:
            continue

        results[f"{group_name}_mae"] = float(
            np.mean(np.abs(group_errors))
        )

        results[f"{group_name}_phm08"] = (
            calculate_phm08(group_errors)
        )

        results[f"{group_name}_over"] = int(
            np.sum(group_errors > 0)
        )

        results[f"{group_name}_under"] = int(
            np.sum(group_errors < 0)
        )

    return results


# ==========================================
# 4. LOAD DATA
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

criterion = RULAwareAsymmetricLoss(
    alpha_low=ALPHA_LOW,
    alpha_medium=ALPHA_MEDIUM,
    alpha_high=ALPHA_HIGH,
    beta=BETA
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n===== MODEL CONFIGURATION =====")

print("Input size:", input_size)
print("Hidden size:", HIDDEN_SIZE)

print("Alpha Low:", ALPHA_LOW)
print("Alpha Medium:", ALPHA_MEDIUM)
print("Alpha High:", ALPHA_HIGH)
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

    train_loss = total_train_loss / train_count


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

    validation_mae = float(
        np.mean(np.abs(errors))
    )

    validation_rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    validation_phm08 = calculate_phm08(
        errors
    )

    group_metrics = calculate_group_metrics(
        errors,
        all_targets
    )


    # --------------------------------------
    # SAVE HISTORY
    # --------------------------------------

    epoch_record = {
        "epoch": epoch + 1,
        "train_rul_aware_loss": train_loss,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_phm08": validation_phm08
    }

    epoch_record.update(group_metrics)

    history.append(epoch_record)


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    # Keep the same checkpoint selection rule
    # as the previous LSTM experiments: lowest MAE.

    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae
        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "input_size": int(input_size),
                "hidden_size": int(HIDDEN_SIZE),

                "alpha_low": float(ALPHA_LOW),
                "alpha_medium": float(ALPHA_MEDIUM),
                "alpha_high": float(ALPHA_HIGH),
                "beta": float(BETA),

                "low_rul_threshold": int(
                    LOW_RUL_THRESHOLD
                ),

                "medium_rul_threshold": int(
                    MEDIUM_RUL_THRESHOLD
                ),

                "best_epoch": int(best_epoch),

                "validation_mae": float(
                    validation_mae
                ),

                "validation_rmse": float(
                    validation_rmse
                ),

                "validation_phm08": float(
                    validation_phm08
                )
            },
            MODEL_PATH
        )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val MAE: {validation_mae:.4f} | "
        f"Val RMSE: {validation_rmse:.4f} | "
        f"Val PHM08: {validation_phm08:.2f} | "
        f"Low PHM08: "
        f"{group_metrics.get('low_phm08', 0):.2f}"
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

print("\n===== RUL-AWARE LSTM TRAINING COMPLETED =====")

print("Best epoch:", best_epoch)

print(
    "Best validation MAE:",
    best_validation_mae
)

print("Model saved:", MODEL_PATH)
print("History saved:", HISTORY_PATH)
