
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "raw" / "cmapss"

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "neural_network.pt"
)

SCALER_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "standard_scaler.joblib"
)

RUL_PATH = DATA_DIR / "RUL_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"

REPORT_DIR = BASE_DIR / "reports"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. DEFINE SAME NEURAL NETWORK ARCHITECTURE
# ==========================================

class RULRegressor(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim_1,
        hidden_dim_2,
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),

            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x):

        return self.network(x)


# ==========================================
# 3. PHM08 SCORING FUNCTION
# ==========================================

def calculate_phm08_score(y_true, y_pred):

    errors = y_pred - y_true

    penalties = np.zeros_like(
        errors,
        dtype=float,
    )

    under_mask = errors < 0
    over_mask = errors >= 0

    penalties[under_mask] = np.expm1(
        -errors[under_mask] / 13.0
    )

    penalties[over_mask] = np.expm1(
        errors[over_mask] / 10.0
    )

    return penalties.sum()


# ==========================================
# 4. MAIN FUNCTION
# ==========================================

def main():

    # --------------------------------------
    # 4.1 LOAD TRAINED CHECKPOINT
    # --------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    input_dim = checkpoint["input_dim"]

    hidden_dim_1 = checkpoint["hidden_dims"][0]

    hidden_dim_2 = checkpoint["hidden_dims"][1]

    feature_columns = checkpoint["feature_columns"]

    model = RULRegressor(
        input_dim,
        hidden_dim_1,
        hidden_dim_2,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("===== NEURAL NETWORK INFORMATION =====")

    print("Best epoch:", checkpoint["best_epoch"])

    print(
        "Best validation MAE:",
        checkpoint["validation_mae"],
    )

    print("Number of features:", input_dim)

    print("Device:", DEVICE)

    # --------------------------------------
    # 4.2 LOAD STANDARD SCALER
    # --------------------------------------

    scaler = joblib.load(SCALER_PATH)

    # --------------------------------------
    # 4.3 LOAD NASA TEST DATA
    # --------------------------------------

    columns = (
        [
            "unit_number",
            "time_cycles",
            "setting_1",
            "setting_2",
            "setting_3",
        ]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    test_df = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    rul_df = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    # --------------------------------------
    # 4.4 GET FINAL CYCLE OF EACH ENGINE
    # --------------------------------------

    test_final = (
        test_df
        .sort_values(
            ["unit_number", "time_cycles"]
        )
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )

    y_test = rul_df["RUL"].to_numpy(
        dtype=np.float32
    )

    if len(test_final) != len(y_test):
        raise ValueError(
            "Test engine count does not match NASA RUL labels!"
        )

    if test_final["unit_number"].astype(int).tolist() != list(
        range(1, len(y_test) + 1)
    ):
        raise ValueError(
            "Engine order does not match NASA RUL labels!"
        )

    print("\n===== TEST DATA INFORMATION =====")

    print("Test engines:", len(test_final))

    # --------------------------------------
    # 4.5 SELECT AND SCALE FEATURES
    # --------------------------------------

    X_test_raw = test_final[feature_columns]

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------
    # 4.6 GENERATE PREDICTIONS
    # --------------------------------------

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test_tensor
        ).cpu().numpy().reshape(-1)

    # --------------------------------------
    # 4.7 CALCULATE METRICS
    # --------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    phm08_score = calculate_phm08_score(
        y_test,
        predictions,
    )

    errors = predictions - y_test

    overprediction = (errors > 0).sum()

    underprediction = (errors < 0).sum()

    # --------------------------------------
    # 4.8 DISPLAY RESULTS
    # --------------------------------------

    print("\n===== NEURAL NETWORK OFFICIAL TEST METRICS =====")

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"PHM08 Score: {phm08_score:.4f}")

    print("Overprediction:", overprediction)

    print("Underprediction:", underprediction)

    # --------------------------------------
    # 4.9 SAVE PREDICTIONS AND METRICS
    # --------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame({
        "unit_number": test_final["unit_number"],
        "final_cycle": test_final["time_cycles"],
        "actual_RUL": y_test,
        "predicted_RUL": predictions,
        "error": errors,
        "absolute_error": np.abs(errors),
    })

    predictions_df.to_csv(
        REPORT_DIR / "neural_network_test_predictions.csv",
        index=False,
    )

    metrics_df = pd.DataFrame([{
        "model": "Neural Network",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PHM08_Score": phm08_score,
        "Overprediction": int(overprediction),
        "Underprediction": int(underprediction),
    }])

    metrics_df.to_csv(
        REPORT_DIR / "neural_network_test_metrics.csv",
        index=False,
    )

    print("\n===== SAMPLE PREDICTIONS =====")

    print(
        predictions_df.head(10).to_string(index=False)
    )

    print("\nNeural Network Test Evaluation Completed!")


if __name__ == "__main__":
    main()
