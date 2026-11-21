
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

BUNDLE_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "final_ml_model_bundle.joblib"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD001.txt"
)

REPORT_DIR = BASE_DIR / "reports"


def main():

    # ==========================================
    # 2. LOAD MODEL BUNDLE
    # ==========================================

    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {BUNDLE_PATH}"
        )

    bundle = joblib.load(BUNDLE_PATH)

    model = bundle["model"]

    scaler = bundle["scaler"]

    feature_columns = bundle["feature_columns"]

    print("===== MODEL BUNDLE INFORMATION =====")
    print("Model:", bundle["model_name"])
    print("Features:", len(feature_columns))

    # ==========================================
    # 3. LOAD RAW NASA TEST DATA
    # ==========================================

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

    # ==========================================
    # 4. SELECT FINAL CYCLE PER ENGINE
    # ==========================================

    test_final = (
        test_df
        .sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )

    X_raw = test_final[feature_columns]

    # ==========================================
    # 5. PREPROCESS RAW INPUT
    # ==========================================

    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        columns=feature_columns,
    )

    # ==========================================
    # 6. GENERATE PREDICTIONS
    # ==========================================

    predictions = model.predict(X_scaled)

    # ==========================================
    # 7. VALIDATE PREDICTIONS
    # ==========================================

    if len(predictions) != len(test_final):
        raise ValueError(
            "Prediction count does not match engine count!"
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain invalid numerical values!"
        )

    prediction_report = pd.DataFrame({
        "unit_number": test_final["unit_number"],
        "final_cycle": test_final["time_cycles"],
        "predicted_RUL": predictions,
    })

    # ==========================================
    # 8. DISPLAY RESULTS
    # ==========================================

    print("\n===== FINAL MODEL SMOKE TEST =====")

    print("Input engines:", len(X_raw))

    print("Prediction count:", len(predictions))

    print("All predictions finite:", np.isfinite(predictions).all())

    print("\nSample predictions:")

    print(
        prediction_report.head(5).to_string(index=False)
    )

    print("\nFinal model verification successful!")

    # ==========================================
    # 9. SAVE SMOKE TEST REPORT
    # ==========================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_report.to_csv(
        REPORT_DIR / "final_model_smoke_test.csv",
        index=False,
    )

    print("\nSmoke test report saved successfully!")


if __name__ == "__main__":
    main()
