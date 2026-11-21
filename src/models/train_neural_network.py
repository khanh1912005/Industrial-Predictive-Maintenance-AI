 
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_FD001_standard_scaled.csv"
)

VALIDATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation_FD001_standard_scaled.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)

MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "neural_network.pt"

HISTORY_PATH = REPORT_DIR / "neural_network_training_history.csv"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32

RANDOM_SEED = 42


# ==========================================
# 2. SET RANDOM SEED AND DEVICE
# ==========================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", DEVICE)


# ==========================================
# 3. DEFINE NEURAL NETWORK
# ==========================================

class RULRegressor(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, HIDDEN_DIM_1),

            nn.ReLU(),

            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),

            nn.ReLU(),

            nn.Linear(HIDDEN_DIM_2, 1),
        )

    def forward(self, x):

        return self.network(x)


# ==========================================
# 4. LOAD DATA
# ==========================================

def load_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    train_df = pd.read_csv(TRAIN_PATH)

    validation_df = pd.read_csv(VALIDATION_PATH)

    required_columns = feature_columns + ["RUL"]

    for df, name in [
        (train_df, "Training"),
        (validation_df, "Validation"),
    ]:

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing columns: {missing}"
            )

    X_train = train_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_train = train_df["RUL"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    X_validation = validation_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_validation = validation_df["RUL"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


# ==========================================
# 5. CREATE DATA LOADERS
# ==========================================

def create_data_loaders(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(X_validation),
        torch.from_numpy(y_validation),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, validation_loader, len(train_dataset)


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def evaluate_model(model, validation_loader, criterion):

    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in validation_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size

            total_absolute_error += (
                torch.abs(predictions - y_batch).sum().item()
            )

            total_samples += batch_size

    validation_mse = total_loss / total_samples

    validation_mae = (
        total_absolute_error / total_samples
    )

    return validation_mse, validation_mae


# ==========================================
# 7. MAIN TRAINING FUNCTION
# ==========================================

def main():

    # --------------------------------------
    # 7.1 Load data
    # --------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = load_data()

    print("\n===== DATA INFORMATION =====")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print("Number of features:", len(feature_columns))

    # --------------------------------------
    # 7.2 Create DataLoaders
    # --------------------------------------

    train_loader, validation_loader, train_size = (
        create_data_loaders(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    print("\nIterations per epoch:", len(train_loader))

    # --------------------------------------
    # 7.3 Initialize model
    # --------------------------------------

    model = RULRegressor(
        input_dim=len(feature_columns)
    ).to(DEVICE)

    # --------------------------------------
    # 7.4 Define Loss and Optimizer
    # --------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------
    # 7.5 Training configuration
    # --------------------------------------

    best_validation_mae = float("inf")

    history = []

    print("\n===== START TRAINING =====")

    # --------------------------------------
    # 7.6 Training loop
    # --------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward propagation
            predictions = model(X_batch)

            # Calculate loss
            loss = criterion(predictions, y_batch)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            total_train_loss += (
                loss.item() * X_batch.size(0)
            )

        train_mse = total_train_loss / train_size

        # ----------------------------------
        # 7.7 Validation
        # ----------------------------------

        validation_mse, validation_mae = evaluate_model(
            model,
            validation_loader,
            criterion,
        )

        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_mae": validation_mae,
        })

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {validation_mse:.4f} | "
            f"Val MAE: {validation_mae:.4f}"
        )

        # ----------------------------------
        # 7.8 Save best model
        # ----------------------------------

        if validation_mae < best_validation_mae:

            best_validation_mae = validation_mae

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": len(feature_columns),
                    "hidden_dims": [
                        HIDDEN_DIM_1,
                        HIDDEN_DIM_2,
                    ],
                    "feature_columns": feature_columns,
                    "best_epoch": epoch + 1,
                    "validation_mae": validation_mae,
                },
                MODEL_PATH,
            )

    # --------------------------------------
    # 7.9 Save training history
    # --------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print("\n===== TRAINING COMPLETED =====")

    print("Best validation MAE:", best_validation_mae)

    print("Best model saved:", MODEL_PATH)

    print("Training history saved:", HISTORY_PATH)


if __name__ == "__main__":
    main()
