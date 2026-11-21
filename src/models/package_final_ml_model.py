
from pathlib import Path
import json

import joblib


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "artifacts" / "models"

SCALER_DIR = BASE_DIR / "artifacts" / "scalers"

REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"

SCALER_PATH = SCALER_DIR / "standard_scaler.joblib"

METADATA_PATH = SCALER_DIR / "feature_metadata.json"

BUNDLE_PATH = MODEL_DIR / "final_ml_model_bundle.joblib"

SELECTION_REPORT_PATH = (
    REPORT_DIR / "final_ml_model_selection.json"
)


# ==========================================
# 2. FINAL MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "Random Forest"

RUL_CAP = 125


def main():

    # ==========================================
    # 3. CHECK REQUIRED FILES
    # ==========================================

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        METADATA_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # ==========================================
    # 4. LOAD FINAL MODEL AND SCALER
    # ==========================================

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    if not feature_columns:
        raise ValueError("Feature list is empty!")

    print("===== FINAL MODEL INFORMATION =====")
    print("Model:", MODEL_NAME)
    print("Number of features:", len(feature_columns))

    # ==========================================
    # 5. CREATE MODEL BUNDLE
    # ==========================================

    model_bundle = {
        "project_name": "Industrial Predictive Maintenance AI",

        "model_name": MODEL_NAME,

        "model": model,

        "scaler": scaler,

        "feature_columns": feature_columns,

        "target_column": "RUL",

        "rul_cap": RUL_CAP,

        "scaling_method": "StandardScaler",

        "scaler_fitted_on": "Training data",

        "input_contract": (
            "Raw sensor feature values. "
            "Features must follow feature_columns order."
        ),
    }

    # ==========================================
    # 6. SAVE MODEL BUNDLE
    # ==========================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_bundle,
        BUNDLE_PATH,
        compress=3,
    )

    # ==========================================
    # 7. SAVE MODEL SELECTION REPORT
    # ==========================================

    selection_report = {
        "project": "Industrial Predictive Maintenance AI",

        "primary_model": "Random Forest",

        "challenger_model": "XGBoost",

        "baseline_model": "Linear Regression",

        "selection_reason": [
            "Lowest PHM08 score on official NASA Test among the three evaluated models.",
            "Lowest RMSE on official NASA Test.",
            "Highest R2 on official NASA Test.",
        ],

        "official_test_metrics": {
            "Random Forest": {
                "MAE": 12.7535,
                "RMSE": 17.5130,
                "R2": 0.8224,
                "PHM08_Total_Score": 720.4138,
            },

            "XGBoost": {
                "MAE": 12.5057,
                "RMSE": 17.7183,
                "R2": 0.8182,
                "PHM08_Total_Score": 818.0580,
            },

            "Linear Regression": {
                "MAE": 17.6182,
                "RMSE": 21.8754,
                "R2": 0.7229,
                "PHM08_Total_Score": 1307.5510,
            },
        },

        "limitations": [
            "Model selection is retrospective because the official Test set was used for model comparison.",
            "The dataset is NASA C-MAPSS simulated turbofan data.",
            "The selected model is a portfolio research model, not a certified industrial safety system.",
        ],
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SELECTION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            selection_report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # ==========================================
    # 8. DISPLAY RESULTS
    # ==========================================

    print("\n===== MODEL PACKAGING COMPLETED =====")

    print("Bundle saved:", BUNDLE_PATH)

    print("Selection report saved:", SELECTION_REPORT_PATH)

    print("\nModel:", MODEL_NAME)

    print("Features:", len(feature_columns))


if __name__ == "__main__":
    main()
