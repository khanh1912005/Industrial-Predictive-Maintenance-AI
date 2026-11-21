
import json
from copy import deepcopy
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD001.txt"
)

API_URL = "http://127.0.0.1:8000/api/v1/predict/rul"
HEALTH_URL = "http://127.0.0.1:8000/health"

ENGINE_ID = 1
SEQUENCE_LENGTH = 30


# ==========================================
# 2. LOAD NASA TEST DATA
# ==========================================

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

engine_df = test_df[
    test_df["unit_number"] == ENGINE_ID
].sort_values("time_cycles")

last_window = engine_df.tail(SEQUENCE_LENGTH)

if len(last_window) != SEQUENCE_LENGTH:
    raise ValueError("Engine does not have 30 cycles.")


# ==========================================
# 3. BUILD VALID REQUEST
# ==========================================

feature_columns = [
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

cycles = []

for _, row in last_window.iterrows():

    cycle_data = {
        "cycle_number": int(row["time_cycles"])
    }

    for feature in feature_columns:
        cycle_data[feature] = float(row[feature])

    cycles.append(cycle_data)


valid_payload = {
    "engine_id": "ENGINE-001",
    "cycles": cycles
}


# ==========================================
# 4. SEND HTTP REQUEST
# ==========================================

def send_post_request(payload):

    request_body = json.dumps(
        payload,
        allow_nan=False
    ).encode("utf-8")

    http_request = Request(
        API_URL,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        method="POST"
    )

    try:

        with urlopen(http_request, timeout=30) as response:

            body = response.read().decode("utf-8")

            return response.status, json.loads(body)

    except HTTPError as error:

        body = error.read().decode("utf-8")

        try:
            response_data = json.loads(body)
        except json.JSONDecodeError:
            response_data = {"detail": body}

        return error.code, response_data

    except URLError as error:

        raise RuntimeError(
            f"Cannot connect to FastAPI: {error.reason}. "
            "Check whether Uvicorn is running."
        ) from error


# ==========================================
# 5. RUN ONE TEST CASE
# ==========================================

def run_test(test_name, payload, expected_status):

    status_code, response_data = send_post_request(payload)

    passed = status_code == expected_status

    if passed:
        print(
            f"[PASS] {test_name} | "
            f"HTTP {status_code}"
        )
    else:
        print(
            f"[FAIL] {test_name} | "
            f"Expected {expected_status}, "
            f"received {status_code}"
        )

        print("Response:", response_data)

    return passed


# ==========================================
# 6. TEST HEALTH ENDPOINT
# ==========================================

def test_health():

    request = Request(
        HEALTH_URL,
        method="GET"
    )

    try:

        with urlopen(request, timeout=10) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            passed = (
                response.status == 200
                and result.get("model_loaded") is True
            )

            if passed:
                print("[PASS] Health check | HTTP 200")
            else:
                print("[FAIL] Health check")
                print(result)

            return passed

    except HTTPError as error:

        print("[FAIL] Health check | HTTP", error.code)
        return False


# ==========================================
# 7. PREPARE INVALID REQUESTS
# ==========================================

# Case 1: Only 29 cycles
payload_29 = deepcopy(valid_payload)
payload_29["cycles"].pop()


# Case 2: Missing sensor_21
payload_missing_sensor = deepcopy(valid_payload)
del payload_missing_sensor["cycles"][0]["sensor_21"]


# Case 3: Unsupported feature
payload_extra_feature = deepcopy(valid_payload)
payload_extra_feature["cycles"][0]["sensor_1"] = 123.0


# Case 4: Non-consecutive cycles
payload_cycle_gap = deepcopy(valid_payload)

# Shift cycles from position 10 onward by one.
# This creates a gap while keeping the sequence increasing.
for cycle in payload_cycle_gap["cycles"][10:]:
    cycle["cycle_number"] += 1


# Case 5: Invalid feature data type
payload_invalid_type = deepcopy(valid_payload)
payload_invalid_type["cycles"][0]["sensor_21"] = "invalid-value"


# ==========================================
# 8. RUN ALL TESTS
# ==========================================

def main():

    results = []

    print("===== API ERROR HANDLING TESTS =====\n")

    results.append(
        test_health()
    )

    results.append(
        run_test(
            "Valid prediction request",
            valid_payload,
            expected_status=200
        )
    )

    results.append(
        run_test(
            "Request with 29 cycles",
            payload_29,
            expected_status=422
        )
    )

    results.append(
        run_test(
            "Missing sensor_21",
            payload_missing_sensor,
            expected_status=422
        )
    )

    results.append(
        run_test(
            "Unsupported sensor_1",
            payload_extra_feature,
            expected_status=422
        )
    )

    results.append(
        run_test(
            "Non-consecutive cycles",
            payload_cycle_gap,
            expected_status=422
        )
    )

    results.append(
        run_test(
            "Invalid sensor data type",
            payload_invalid_type,
            expected_status=422
        )
    )


    # ======================================
    # 9. TEST SUMMARY
    # ======================================

    passed = sum(results)
    total = len(results)

    print("\n===== API TEST SUMMARY =====")

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed != total:
        raise AssertionError("Some API tests failed.")

    print("\nAll API tests passed successfully.")


if __name__ == "__main__":
    main()
