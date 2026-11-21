
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from src.models.lstm_model import LSTMRegressor


# ==========================================
# 1. MODEL ARTIFACT CONTAINER
# ==========================================

@dataclass
class ModelArtifacts:

    model: LSTMRegressor
    scaler: Any
    feature_columns: list[str]
    config: dict
    device: torch.device


# ==========================================
# 2. LOAD MODEL PACKAGE
# ==========================================

def load_model_package(
    package_dir: Path | None = None
) -> ModelArtifacts:

    # --------------------------------------
    # Resolve package directory
    # --------------------------------------

    if package_dir is None:

        base_dir = Path(__file__).resolve().parents[2]

        package_dir = (
            base_dir
            / "artifacts"
            / "final_dl_model"
        )

    package_dir = Path(package_dir)


    # --------------------------------------
    # Define package files
    # --------------------------------------

    model_path = package_dir / "lstm.pt"

    scaler_path = (
        package_dir / "standard_scaler.joblib"
    )

    metadata_path = (
        package_dir / "feature_metadata.json"
    )

    config_path = (
        package_dir / "model_config.json"
    )

    required_files = [
        model_path,
        scaler_path,
        metadata_path,
        config_path
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Model package file not found: {file_path}"
            )


    # ======================================
    # 3. LOAD CONFIGURATION AND METADATA
    # ======================================

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)


    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    if feature_columns != config["feature_columns"]:

        raise ValueError(
            "Feature metadata does not match model configuration."
        )


    # ======================================
    # 4. LOAD STANDARD SCALER
    # ======================================

    scaler = joblib.load(scaler_path)

    if scaler.n_features_in_ != len(feature_columns):

        raise ValueError(
            "Scaler feature count does not match metadata."
        )

    if scaler.__class__.__name__ != config["scaler_type"]:

        raise ValueError(
            "Scaler type does not match model configuration."
        )


    # ======================================
    # 5. LOAD PYTORCH CHECKPOINT
    # ======================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    input_size = int(checkpoint["input_size"])
    hidden_size = int(checkpoint["hidden_size"])

    if input_size != len(feature_columns):

        raise ValueError(
            "LSTM input size does not match feature count."
        )

    if (
        input_size != int(config["input_size"])
        or hidden_size != int(config["hidden_size"])
    ):

        raise ValueError(
            "Checkpoint configuration does not match model package."
        )


    # ======================================
    # 6. RECONSTRUCT LSTM MODEL
    # ======================================

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Switch to inference mode
    model.eval()


    # ======================================
    # 7. RETURN LOADED ARTIFACTS
    # ======================================

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        config=config,
        device=device
    )
