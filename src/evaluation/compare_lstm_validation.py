from json.tool import main
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

WINDOW_DIR = BASE_DIR / "data" / "processed" / "windows"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "reports"

BASELINE_PATH = MODEL_DIR / "lstm.pt"
ASYMMETRIC_PATH = (
    MODEL_DIR / "lstm_asymmetric_alpha3_beta1.pt"
)

OVERALL_REPORT_PATH = (
    REPORT_DIR / "lstm_validation_comparison.csv"
)
GROUP_REPORT_PATH = (
    REPORT_DIR / "lstm_validation_comparison_by_rul.csv"
)

BATCH_SIZE = 128
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. LOAD AND VALIDATE CHECKPOINT SAFELY
# ==========================================

def load_trusted_checkpoint(checkpoint_path):
    """Load a checkpoint created by this project.

    The asymmetric training script may save NumPy scalar metadata
    (for example, np.float64 validation metrics). We explicitly
    allowlist only the NumPy scalar/dtype objects needed for those
    metadata values while keeping weights_only=True.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # NumPy 2.x exposes _core; older NumPy versions expose core.
    try:
        numpy_scalar = np._core.multiarray.scalar
    except AttributeError:
        numpy_scalar = np.core.multiarray.scalar

    allowed_numpy_objects = [
        numpy_scalar,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]

    # This context manager is available in the PyTorch version
    # identified by the user's error message (PyTorch 2.6+).
    with torch.serialization.safe_globals(allowed_numpy_objects):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

    required_keys = {
        "model_state_dict",
        "input_size",
        "hidden_size",
        "best_epoch",
    }

    missing_keys = required_keys.difference(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}"
        )

    return checkpoint


# ==========================================
# 3. LOAD VALIDATION DATA
# ==========================================

if not WINDOW_DIR.exists():
    raise FileNotFoundError(
        f"Window directory not found: {WINDOW_DIR}"
    )

X_validation = np.load(
    WINDOW_DIR / "X_validation.npy"
).astype(np.float32)

y_validation = np.load(
    WINDOW_DIR / "y_validation.npy"
).astype(np.float32)

if X_validation.ndim != 3:
    raise ValueError(
        f"Expected X_validation to be 3D, got shape {X_validation.shape}"
    )

if y_validation.ndim != 1:
    raise ValueError(
        f"Expected y_validation to be 1D, got shape {y_validation.shape}"
    )

if len(X_validation) != len(y_validation):
    raise ValueError(
        "X_validation and y_validation have different sample counts."
    )

if not np.isfinite(X_validation).all():
    raise ValueError("X_validation contains NaN or Inf values.")

if not np.isfinite(y_validation).all():
    raise ValueError("y_validation contains NaN or Inf values.")

X_tensor = torch.from_numpy(X_validation).to(DEVICE)

print("Device:", DEVICE)
print("Validation X:", X_validation.shape)
print("Validation y:", y_validation.shape)


# ==========================================
# 4. PHM08 METRIC
# ==========================================

def calculate_phm08(errors):
    """Calculate the project PHM08 asymmetric score.

    Positive error: overprediction, penalty denominator = 10.
    Negative error: underprediction, penalty denominator = 13.
    Lower total score is better.
    """

    errors = np.asarray(errors, dtype=np.float64)

    scores = np.where(
        errors < 0,
        np.expm1(-errors / 13.0),
        np.expm1(errors / 10.0),
    )

    return float(scores.sum())


# ==========================================
# 5. EVALUATE ONE CHECKPOINT
# ==========================================

def evaluate_model(checkpoint_path, model_name):
    checkpoint = load_trusted_checkpoint(checkpoint_path)

    input_size = int(checkpoint["input_size"])
    hidden_size = int(checkpoint["hidden_size"])

    if input_size != X_validation.shape[2]:
        raise ValueError(
            f"{model_name}: checkpoint expects {input_size} features, "
            f"but validation data has {X_validation.shape[2]}."
        )

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate predictions in batches to avoid unnecessary memory use.
    prediction_batches = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), BATCH_SIZE):
            batch_x = X_tensor[start:start + BATCH_SIZE]
            batch_predictions = model(batch_x)

            prediction_batches.append(
                batch_predictions.cpu().numpy()
            )

    predictions = np.concatenate(prediction_batches).astype(np.float64)
    targets = y_validation.astype(np.float64)
    errors = predictions - targets

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"{model_name} produced NaN or Inf predictions."
        )

    overall_results = {
        "Model": model_name,
        "Best_Epoch": int(checkpoint["best_epoch"]),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "PHM08": calculate_phm08(errors),
        "Mean_Error": float(np.mean(errors)),
        "Overprediction": int(np.sum(errors > 0)),
        "Underprediction": int(np.sum(errors < 0)),
        "Exact_Prediction": int(np.sum(errors == 0)),
    }

    # Use the same RUL boundaries as the project's error analysis:
    # Low <= 30, Medium (30, 80], High > 80.
    rul_groups = pd.cut(
        targets,
        bins=[-np.inf, 30, 80, np.inf],
        labels=["Low", "Medium", "High"],
        right=True,
    )

    group_results = []

    for group_name in ["Low", "Medium", "High"]:
        mask = np.asarray(rul_groups == group_name)
        group_errors = errors[mask]

        if group_errors.size == 0:
            continue

        group_results.append({
            "Model": model_name,
            "RUL_Group": group_name,
            "Count": int(group_errors.size),
            "MAE": float(np.mean(np.abs(group_errors))),
            "RMSE": float(np.sqrt(np.mean(group_errors ** 2))),
            "PHM08": calculate_phm08(group_errors),
            "Mean_Error": float(np.mean(group_errors)),
            "Overprediction": int(np.sum(group_errors > 0)),
            "Underprediction": int(np.sum(group_errors < 0)),
        })

    return overall_results, group_results


# ==========================================
# 6. COMPARE BOTH MODELS
# ==========================================

models_to_evaluate = [
    (BASELINE_PATH, "LSTM-Capped"),
    (ASYMMETRIC_PATH, "LSTM-Asymmetric"),
    (
        MODEL_DIR / "lstm_rul_aware.pt",
        "LSTM-RUL-Aware"
    )
]
overall_results = []
group_results = []

for checkpoint_path, model_name in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")

        results, groups = evaluate_model(
            checkpoint_path,
            model_name,
        )

        overall_results.append(results)
        group_results.extend(groups)

overall_df = pd.DataFrame(overall_results)
group_df = pd.DataFrame(group_results)

print("\n===== OVERALL VALIDATION COMPARISON =====")
print(overall_df.round(4).to_string(index=False))

print("\n===== VALIDATION BY RUL GROUP =====")
print(group_df.round(4).to_string(index=False))

REPORT_DIR.mkdir(parents=True, exist_ok=True)

overall_df.to_csv(
        OVERALL_REPORT_PATH,
        index=False,
    )

group_df.to_csv(
        GROUP_REPORT_PATH,
        index=False,
    )

print("\n===== REPORTS SAVED =====")
print(OVERALL_REPORT_PATH)
print(GROUP_REPORT_PATH)


if __name__ == "__main__":
    main()
