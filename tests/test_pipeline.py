import numpy as np

from src.pipeline import InferencePipeline


def test_pipeline():
    pipeline = InferencePipeline()
    dummy_input = np.random.rand(128)
    result = pipeline.predict(dummy_input)

    assert "gas" in result
    assert 0 <= result["confidence"] <= 1
    assert result["drift_risk"] is False or result["drift_risk"] is True
    assert isinstance(result["concentration_ppm"], float)
