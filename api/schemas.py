from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    readings: list[float] = Field(min_length=128, max_length=128)
    model: Literal["svm", "rf"] = "svm"
    compensate_drift: bool = True


class PredictResponse(BaseModel):
    gas: str
    confidence: float
    concentration_ppm: float
    drift_risk: bool
    compensated: bool
    model_used: str
