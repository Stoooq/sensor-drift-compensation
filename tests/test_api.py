from fastapi.testclient import TestClient

from api.main import app
from api.schemas import PredictResponse


def test_api():
    with TestClient(app) as client:
        response = client.post("/predict", json={"readings": [0.5] * 128})

    assert response.status_code == 200
    assert PredictResponse(**response.json())
