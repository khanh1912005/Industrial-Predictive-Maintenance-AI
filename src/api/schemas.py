
from math import isfinite
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator
)


# ==========================================
# 1. ENGINE ID
# ==========================================

EngineId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64
    )
]


# ==========================================
# 2. ONE CYCLE: 17 FEATURES
# ==========================================

class CycleFeatures(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    # Metadata, not an input feature
    cycle_number: int = Field(ge=1)

    # Two operational settings
    setting_1: float
    setting_2: float

    # Fifteen selected sensors
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float

    @field_validator("*")
    @classmethod
    def validate_finite_values(cls, value):

        if isinstance(value, (int, float)):
            if not isfinite(value):
                raise ValueError(
                    "Feature values must be finite numbers."
                )

        return value


# ==========================================
# 3. PREDICTION REQUEST
# ==========================================

class RULPredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    engine_id: EngineId

    cycles: list[CycleFeatures] = Field(
        min_length=30,
        max_length=30
    )

    @model_validator(mode="after")
    def validate_cycle_order(self):

        cycle_numbers = [
            cycle.cycle_number
            for cycle in self.cycles
        ]

        for previous, current in zip(
            cycle_numbers,
            cycle_numbers[1:]
        ):

            if current <= previous:
                raise ValueError(
                    "Cycles must be ordered chronologically "
                    "with strictly increasing cycle_number."
                )

        return self


# ==========================================
# 4. PREDICTION RESPONSE
# ==========================================

class RULPredictionResponse(BaseModel):

    engine_id: str

    predicted_rul: float

    unit: Literal["cycles"] = "cycles"

    model: str = "LSTM-Capped"

    status: Literal["success"] = "success"
