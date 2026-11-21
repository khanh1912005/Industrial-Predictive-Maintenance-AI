import time
import logging
from contextlib import asynccontextmanager
from src.api.logging_config import configure_logging
from fastapi import (
    FastAPI,
    HTTPException,
    Request as FastAPIRequest
)

from src.api.schemas import (
    RULPredictionRequest,
    RULPredictionResponse
)

from src.services.model_loader import load_model_package

from src.services.prediction_service import predict_rul

configure_logging()

logger = logging.getLogger(__name__)
# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

PROJECT_NAME = "Industrial Predictive Maintenance AI"
PROJECT_VERSION = "0.1.0"

logger = logging.getLogger(__name__)


# ==========================================
# 2. APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting application...")

    # Load model package once during startup
    artifacts = load_model_package()

    app.state.model_artifacts = artifacts

    logger.info(
        "LSTM model loaded successfully."
    )

    yield

    # Release application reference during shutdown
    app.state.model_artifacts = None

    logger.info("Application shutdown completed.")


# ==========================================
# 3. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title=PROJECT_NAME,
    description=(
        "AI-powered Predictive Maintenance API "
        "for Remaining Useful Life prediction."
    ),
    version=PROJECT_VERSION,
    lifespan=lifespan
)

# ==========================================
# HTTP REQUEST LOGGING MIDDLEWARE
# ==========================================

@app.middleware("http")
async def log_http_requests(
    request: FastAPIRequest,
    call_next
):

    start_time = time.perf_counter()

    try:

        response = await call_next(request)

    except Exception:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "HTTP request failed | method=%s | "
            "path=%s | duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms
        )

        raise

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "HTTP request | method=%s | "
        "path=%s | status_code=%s | duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )

    return response


# ==========================================
# 4. ROOT ENDPOINT
# ==========================================

@app.get("/", tags=["System"])
def root():

    return {
        "message": (
            "Welcome to Industrial Predictive Maintenance AI"
        ),
        "documentation": "/docs",
        "health": "/health"
    }


# ==========================================
# 5. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health", tags=["System"])
def health_check(request: FastAPIRequest):

    artifacts = getattr(
        request.app.state,
        "model_artifacts",
        None
    )

    return {
        "status": "ok",
        "model_loaded": artifacts is not None,
        "service": "industrial-predictive-maintenance-api",
        "version": PROJECT_VERSION
    }


# ==========================================
# 6. ABOUT ENDPOINT
# ==========================================

@app.get("/about", tags=["System"])
def about():

    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "model_candidate": "LSTM-Capped",
        "framework": "FastAPI + PyTorch",
        "sequence_length": 30,
        "feature_count": 17
    }


# ==========================================
# 7. RUL PREDICTION ENDPOINT
# ==========================================

@app.post(
    "/api/v1/predict/rul",
    response_model=RULPredictionResponse,
    tags=["Prediction"]
)
def predict_rul_endpoint(
    payload: RULPredictionRequest,
    request: FastAPIRequest
):

    # Get already-loaded model artifacts
    artifacts = getattr(
        request.app.state,
        "model_artifacts",
        None
    )

    if artifacts is None:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not ready."
        )

    try:

        # Call Prediction Service
        result = predict_rul(
            request=payload,
            artifacts=artifacts
        )

        return result

    except ValueError as exc:

        # Invalid input sequence
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:

        # Internal model inference failure
        logger.exception(
            "RUL prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error."
        ) from exc
