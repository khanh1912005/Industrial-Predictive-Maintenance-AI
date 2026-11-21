
import numpy as np
import pandas as pd
import torch

from src.services.model_loader import ModelArtifacts
from src.api.schemas import (
    RULPredictionRequest,
    RULPredictionResponse
)


# ==========================================
# 1. PREDICTION SERVICE
# ==========================================

def predict_rul(
    request: RULPredictionRequest,
    artifacts: ModelArtifacts
) -> RULPredictionResponse:

    # --------------------------------------
    # 1.1 Check sequence length
    # --------------------------------------

    expected_length = int(
        artifacts.config["sequence_length"]
    )

    if len(request.cycles) != expected_length:

        raise ValueError(
            f"Expected {expected_length} cycles, "
            f"received {len(request.cycles)}."
        )


    # --------------------------------------
    # 1.2 Check consecutive cycle numbers
    # --------------------------------------

    cycle_numbers = [
        cycle.cycle_number
        for cycle in request.cycles
    ]

    for previous, current in zip(
        cycle_numbers,
        cycle_numbers[1:]
    ):

        if current != previous + 1:

            raise ValueError(
                "Cycle numbers must be consecutive. "
                f"Found {previous} followed by {current}."
            )


    # --------------------------------------
    # 1.3 Extract features in correct order
    # --------------------------------------

    feature_columns = artifacts.feature_columns

    feature_rows = []

    for cycle in request.cycles:

        row = {
            feature: getattr(cycle, feature)
            for feature in feature_columns
        }

        feature_rows.append(row)


    features_df = pd.DataFrame(
        feature_rows,
        columns=feature_columns
    )


    # --------------------------------------
    # 1.4 Validate raw feature values
    # --------------------------------------

    raw_features = features_df.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(raw_features).all():

        raise ValueError(
            "Sensor features contain NaN or Inf."
        )

    expected_shape = (
        expected_length,
        len(feature_columns)
    )

    if raw_features.shape != expected_shape:

        raise ValueError(
            f"Unexpected feature shape: {raw_features.shape}"
        )


    # --------------------------------------
    # 1.5 Apply StandardScaler
    # --------------------------------------

    scaled_features = artifacts.scaler.transform(
        features_df
    )

    scaled_features = scaled_features.astype(
        np.float32
    )

    if not np.isfinite(scaled_features).all():

        raise ValueError(
            "Scaled features contain NaN or Inf."
        )


    # --------------------------------------
    # 1.6 Convert to PyTorch Tensor
    # --------------------------------------

    input_tensor = torch.from_numpy(
        scaled_features
    ).unsqueeze(0).to(artifacts.device)

    expected_tensor_shape = (
        1,
        expected_length,
        len(feature_columns)
    )

    if tuple(input_tensor.shape) != expected_tensor_shape:

        raise ValueError(
            f"Unexpected tensor shape: {tuple(input_tensor.shape)}"
        )


    # --------------------------------------
    # 1.7 Generate RUL Prediction
    # --------------------------------------

    artifacts.model.eval()

    try:

        with torch.inference_mode():

            prediction = artifacts.model(
                input_tensor
            )

        raw_predicted_rul = float(
            prediction.item()
        )

    except Exception as exc:

        raise RuntimeError(
            "LSTM inference failed."
        ) from exc


    # --------------------------------------
    # 1.8 Validate Prediction
    # --------------------------------------

    if not np.isfinite(raw_predicted_rul):

        raise RuntimeError(
            "Model returned a non-finite RUL prediction."
        )


    # RUL cannot be negative.
    # Apply a non-negative output constraint.
    predicted_rul = max(
        0.0,
        raw_predicted_rul
    )


    # --------------------------------------
    # 1.9 Build API Response
    # --------------------------------------

    return RULPredictionResponse(
        engine_id=request.engine_id,
        predicted_rul=predicted_rul,
        unit="cycles",
        model=artifacts.config["model_name"],
        status="success"
    )
