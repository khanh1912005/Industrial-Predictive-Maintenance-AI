
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import RULPredictionRequest
from src.services.model_loader import load_model_package
from src.services.prediction_service import predict_rul


# ==========================================
# 1. LOAD MODEL PACKAGE
# ==========================================

artifacts = load_model_package()


# ==========================================
# 2. LOAD NASA TEST DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD001.txt"
)

raw_columns = (
    ["unit_number", "time_cycles"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    TEST_PATH,
    sep=r"\s+",
    header=None,
    names=raw_columns
)


# ==========================================
# 3. EXTRACT ENGINE 1 LAST 30 CYCLES
# ==========================================

engine_df = test_df[
    test_df["unit_number"] == 1
].sort_values("time_cycles")

last_window = engine_df.tail(30)

if len(last_window) != 30:
    raise ValueError(
        "Engine 1 does not have enough cycles."
    )


# ==========================================
# 4. BUILD API REQUEST
# ==========================================

cycles = []

for _, row in last_window.iterrows():

    cycle_data = {
        "cycle_number": int(row["time_cycles"])
    }

    for feature in artifacts.feature_columns:

        cycle_data[feature] = float(
            row[feature]
        )

    cycles.append(cycle_data)


request = RULPredictionRequest(
    engine_id="ENGINE-001",
    cycles=cycles
)


# ==========================================
# 5. RUN PREDICTION SERVICE
# ==========================================

response = predict_rul(
    request=request,
    artifacts=artifacts
)


# ==========================================
# 6. VERIFY RESULT
# ==========================================

print("\n===== PREDICTION SERVICE TEST =====")

print("Engine:", response.engine_id)
print("Predicted RUL:", round(response.predicted_rul, 4))
print("Model:", response.model)
print("Status:", response.status)

assert np.isfinite(response.predicted_rul)
assert response.status == "success"

print("\n[PASS] Prediction Service works correctly.")


# ==========================================
# 7. TEST NON-CONSECUTIVE CYCLES
# ==========================================

invalid_cycles = [
    dict(cycle)
    for cycle in cycles
]

# Simulate a missing cycle at the end
invalid_cycles[-1]["cycle_number"] += 1

invalid_request = RULPredictionRequest(
    engine_id="ENGINE-INVALID",
    cycles=invalid_cycles
)

try:

    predict_rul(
        request=invalid_request,
        artifacts=artifacts
    )

except ValueError as error:

    print("\n[PASS] Non-consecutive cycles rejected.")
    print("Reason:", error)

else:

    raise AssertionError(
        "Non-consecutive cycles were accepted."
    )
