import numpy as np
from fastapi import APIRouter, Request

from api.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict")
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    pipeline = request.app.state.pipeline

    result = pipeline.predict(np.array(body.readings))

    return PredictResponse(**result, compensated=True, model_used="svm")
