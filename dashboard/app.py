
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
import os

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "test_FD001.txt"
)

API_BASE_URL_DEFAULT = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
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

REQUIRED_COLUMNS = [
    "cycle_number",
    *FEATURE_COLUMNS
]


# ==========================================
# 2. LOAD NASA DEMO DATA
# ==========================================

def load_demo_data():

    if not RAW_TEST_PATH.exists():

        raise FileNotFoundError(
            "NASA test dataset was not found."
        )

    raw_columns = (
        ["unit_number", "time_cycles"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    test_df = pd.read_csv(
        RAW_TEST_PATH,
        sep=r"\s+",
        header=None,
        names=raw_columns
    )

    engine_df = test_df[
        test_df["unit_number"] == 1
    ].sort_values("time_cycles")

    last_window = engine_df.tail(
        SEQUENCE_LENGTH
    )

    if len(last_window) != SEQUENCE_LENGTH:

        raise ValueError(
            "Engine 1 does not have enough cycles."
        )

    demo_df = last_window[
        ["time_cycles", *FEATURE_COLUMNS]
    ].copy()

    demo_df = demo_df.rename(
        columns={"time_cycles": "cycle_number"}
    )

    return demo_df.reset_index(drop=True)


# ==========================================
# 3. VALIDATE UPLOADED CSV
# ==========================================

def validate_dataframe(df):

    # Remove unnecessary spaces in column names
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        return None, (
            f"Missing columns: {missing_columns}"
        )

    # Keep only the columns required by the model
    df = df[REQUIRED_COLUMNS].copy()

    # Check number of cycles
    if len(df) != SEQUENCE_LENGTH:

        return None, (
            f"Expected {SEQUENCE_LENGTH} cycles, "
            f"received {len(df)}."
        )

    # Convert all required values to numeric
    try:

        for column in REQUIRED_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )

    except (ValueError, TypeError):

        return None, (
            "All cycle numbers and sensor features "
            "must contain numeric values."
        )

    # Check NaN and Inf
    values = df.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():

        return None, (
            "CSV contains missing or non-finite values."
        )

    # Cycle number must be an integer
    cycle_numbers = df["cycle_number"].to_numpy(
        dtype=np.float64
    )

    if not np.equal(
        cycle_numbers,
        np.floor(cycle_numbers)
    ).all():

        return None, (
            "Cycle numbers must be integers."
        )

    cycle_numbers = cycle_numbers.astype(int)

    # Check consecutive cycles
    if not np.all(np.diff(cycle_numbers) == 1):

        return None, (
            "Cycle numbers must be consecutive "
            "and increasing."
        )

    df["cycle_number"] = cycle_numbers

    return df, None


# ==========================================
# 4. DASHBOARD HEADER
# ==========================================

st.title("⚙️ Industrial Predictive Maintenance")

st.caption(
    "AI-powered Remaining Useful Life prediction "
    "using LSTM and NASA C-MAPSS data."
)

st.divider()


# ==========================================
# 5. SIDEBAR CONFIGURATION
# ==========================================

with st.sidebar:

    st.header("System Configuration")

    api_base_url = st.text_input(
        "FastAPI Base URL",
        value=API_BASE_URL_DEFAULT
    )

    engine_id = st.text_input(
        "Engine ID",
        value="ENGINE-001"
    )

    st.divider()

    st.subheader("Backend Status")

    if st.button(
        "Check API Connection",
        key="check_api"
    ):

        try:

            response = requests.get(
                f"{api_base_url.rstrip('/')}/health",
                timeout=5
            )

            if response.status_code == 200:

                health_data = response.json()

                if health_data.get("model_loaded") is True:

                    st.success(
                        "Backend and LSTM are ready."
                    )

                else:

                    st.warning(
                        "Backend is running, but model is not loaded."
                    )

            else:

                st.error(
                    f"Backend returned HTTP {response.status_code}"
                )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Cannot connect to backend: {error}"
            )

    st.divider()

    st.caption("Model: LSTM-Capped")
    st.caption("Sequence Length: 30")
    st.caption("Features: 17")


# ==========================================
# 6. DATA UPLOAD
# ==========================================

st.header("1. Sensor Data Input")

st.write(
    "Upload a CSV containing 30 consecutive cycles "
    "and the 17 selected features."
)

uploaded_file = st.file_uploader(
    "Upload Engine Sensor CSV",
    type=["csv"],
    help=(
        "CSV must contain cycle_number and "
        "17 selected features."
    )
)

if st.button(
    "Load NASA Engine 1 Demo",
    key="load_demo"
):

    try:

        st.session_state["demo_data"] = load_demo_data()

        st.session_state["demo_loaded"] = True

        st.success(
            "NASA Engine 1 demo data loaded."
        )

    except Exception as error:

        st.error(str(error))


# ==========================================
# 7. SELECT DATA SOURCE
# ==========================================

input_df = None
data_source = None

if uploaded_file is not None:

    try:

        input_df = pd.read_csv(uploaded_file)

        data_source = uploaded_file.name

    except Exception as error:

        st.error(
            f"Could not read CSV: {error}"
        )

elif "demo_data" in st.session_state:

    input_df = st.session_state["demo_data"].copy()

    data_source = "NASA Engine 1 Demo"


# ==========================================
# 8. VALIDATE AND DISPLAY DATA
# ==========================================

valid_data = False
validated_df = None

if input_df is not None:

    validated_df, validation_error = validate_dataframe(
        input_df.copy()
    )

    if validation_error:

        st.error(validation_error)

    else:

        valid_data = True

        st.success(
            f"Valid input data: {SEQUENCE_LENGTH} cycles "
            f"and {len(FEATURE_COLUMNS)} features."
        )

        st.caption(
            f"Data source: {data_source}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Input Cycles",
                len(validated_df)
            )

        with col2:

            st.metric(
                "Features",
                len(FEATURE_COLUMNS)
            )

        with col3:

            st.metric(
                "Start Cycle",
                int(validated_df["cycle_number"].iloc[0])
            )

        st.subheader("Sensor Data Preview")

        st.dataframe(
            validated_df,
            use_container_width=True
        )

        # ----------------------------------
        # SENSOR TREND
        # ----------------------------------

        
        # ----------------------------------
        # SENSOR TREND
        # ----------------------------------

        st.subheader("Sensor Monitoring")

        sensor_options = [
            feature
            for feature in FEATURE_COLUMNS
            if feature.startswith("sensor_")
        ]

        selected_sensor = st.selectbox(
            "Select sensor to visualize",
            sensor_options
        )

        sensor_values = validated_df[
            selected_sensor
        ]

        # Calculate sensor statistics
        latest_value = float(sensor_values.iloc[-1])
        minimum_value = float(sensor_values.min())
        maximum_value = float(sensor_values.max())
        mean_value = float(sensor_values.mean())

        first_value = float(sensor_values.iloc[0])

        change_value = latest_value - first_value

        # Display sensor metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Latest Value",
                f"{latest_value:.3f}"
            )

        with col2:
            st.metric(
                "Minimum",
                f"{minimum_value:.3f}"
            )

        with col3:
            st.metric(
                "Maximum",
                f"{maximum_value:.3f}"
            )

        with col4:
            st.metric(
                "Average",
                f"{mean_value:.3f}"
            )

        # Display sensor trend
        st.subheader(f"{selected_sensor} Trend")

        chart_df = validated_df.set_index(
            "cycle_number"
        )[[selected_sensor]]

        st.line_chart(
            chart_df,
            use_container_width=True
        )

        # Display overall change
        st.caption(
            f"Change from first to last cycle: "
            f"{change_value:+.3f}"
        )

        st.caption(
            "Sensor statistics describe the observed window. "
            "They are not independent failure thresholds."
        )

        # ----------------------------------
        # DOWNLOAD VALIDATED CSV
        # ----------------------------------

        csv_data = validated_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Validated CSV",
            data=csv_data,
            file_name="validated_engine_data.csv",
            mime="text/csv"
        )


# ==========================================
# 9. PREDICT RUL
# ==========================================

st.divider()

st.header("2. RUL Prediction")

st.write(
    "Send the validated sensor sequence to FastAPI "
    "and receive the LSTM prediction."
)

if st.button(
    "Predict RUL",
    type="primary",
    disabled=(
        not valid_data
        or not engine_id.strip()
    )
):

    # Build API payload
    cycles = []

    for _, row in validated_df.iterrows():

        cycle_data = {
            "cycle_number": int(row["cycle_number"])
        }

        for feature in FEATURE_COLUMNS:

            cycle_data[feature] = float(
                row[feature]
            )

        cycles.append(cycle_data)

    payload = {
        "engine_id": engine_id.strip(),
        "cycles": cycles
    }

    api_url = (
        f"{api_base_url.rstrip('/')}"
        "/api/v1/predict/rul"
    )

    try:

        with st.spinner(
            "LSTM is processing sensor data..."
        ):

            response = requests.post(
                api_url,
                json=payload,
                timeout=60
            )

        if response.status_code == 200:

            result = response.json()

            st.session_state["prediction_result"] = result

        else:

            try:

                error_data = response.json()

            except ValueError:

                error_data = {
                    "detail": response.text
                }

            st.error(
                f"API Error {response.status_code}: "
                f"{error_data.get('detail', error_data)}"
            )

    except requests.exceptions.Timeout:

        st.error(
            "Prediction request timed out."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please check whether the backend is running."
        )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Request failed: {error}"
        )


# ==========================================
# 10. DISPLAY PREDICTION RESULT
# ==========================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.header("3. Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Engine ID",
            result["engine_id"]
        )

    with col2:

        st.metric(
            "Predicted RUL",
            f"{result['predicted_rul']:.2f} cycles"
        )

    with col3:

        st.metric(
            "Model",
            result["model"]
        )

    st.success(
        "RUL prediction completed successfully."
    )

    st.caption(
        "This is a model estimate, not a guaranteed "
        "physical failure time."
    )

    st.json(result)


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Industrial Predictive Maintenance AI | "
    "Powered by FastAPI, PyTorch and Streamlit."
)
