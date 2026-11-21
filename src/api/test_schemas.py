
from copy import deepcopy

from pydantic import ValidationError

from src.api.schemas import RULPredictionRequest


# ==========================================
# 1. SAMPLE SENSOR FEATURES
# ==========================================

SAMPLE_FEATURES = {
    "setting_1": 0.002,
    "setting_2": 0.0003,
    "sensor_2": 642.1,
    "sensor_3": 1589.7,
    "sensor_4": 1400.2,
    "sensor_6": 21.6,
    "sensor_7": 553.4,
    "sensor_8": 2388.1,
    "sensor_9": 9046.2,
    "sensor_11": 47.2,
    "sensor_12": 521.3,
    "sensor_13": 2388.0,
    "sensor_14": 8138.6,
    "sensor_15": 8.4,
    "sensor_17": 392.0,
    "sensor_20": 39.0,
    "sensor_21": 23.4
}


# ==========================================
# 2. BUILD VALID REQUEST
# ==========================================

def create_valid_payload():

    cycles = []

    for cycle_number in range(1, 31):

        cycle_data = {
            "cycle_number": cycle_number,
            **SAMPLE_FEATURES
        }

        cycles.append(cycle_data)

    return {
        "engine_id": "ENGINE-001",
        "cycles": cycles
    }


# ==========================================
# 3. VALIDATION TEST FUNCTION
# ==========================================

def run_test(test_name, payload, expected_valid):

    try:

        RULPredictionRequest.model_validate(payload)

    except ValidationError as error:

        if expected_valid:

            print(f"[FAIL] {test_name}")
            print(error)

            return False

        print(f"[PASS] {test_name}")
        print("Request rejected as expected.")

        return True

    else:

        if not expected_valid:

            print(f"[FAIL] {test_name}")
            print("Invalid request was accepted.")

            return False

        print(f"[PASS] {test_name}")
        print("Request accepted successfully.")

        return True


# ==========================================
# 4. TEST CASES
# ==========================================

def main():

    results = []


    # Test 1: Valid request
    valid_payload = create_valid_payload()

    results.append(
        run_test(
            "Valid request: 30 cycles, 17 features",
            valid_payload,
            expected_valid=True
        )
    )


    # Test 2: Only 29 cycles
    payload_29_cycles = deepcopy(valid_payload)

    payload_29_cycles["cycles"].pop()

    results.append(
        run_test(
            "Invalid request: only 29 cycles",
            payload_29_cycles,
            expected_valid=False
        )
    )


    # Test 3: Missing sensor_21
    payload_missing_sensor = deepcopy(valid_payload)

    del payload_missing_sensor["cycles"][0]["sensor_21"]

    results.append(
        run_test(
            "Invalid request: missing sensor_21",
            payload_missing_sensor,
            expected_valid=False
        )
    )


    # Test 4: NaN value
    payload_nan = deepcopy(valid_payload)

    payload_nan["cycles"][0]["sensor_21"] = float("nan")

    results.append(
        run_test(
            "Invalid request: NaN sensor value",
            payload_nan,
            expected_valid=False
        )
    )


    # Test 5: Incorrect cycle order
    payload_wrong_order = deepcopy(valid_payload)

    payload_wrong_order["cycles"][29]["cycle_number"] = 29

    results.append(
        run_test(
            "Invalid request: incorrect cycle order",
            payload_wrong_order,
            expected_valid=False
        )
    )


    # ======================================
    # 5. TEST SUMMARY
    # ======================================

    passed = sum(results)
    total = len(results)

    print("\n===== VALIDATION TEST SUMMARY =====")

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed != total:

        raise AssertionError(
            "Some validation tests failed."
        )

    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()
