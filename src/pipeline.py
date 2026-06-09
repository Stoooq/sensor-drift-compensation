from pathlib import Path

import joblib
import numpy as np

from src.config import load_config

cfg = load_config(path=Path("config.yaml"))


class InferencePipeline:
    def __init__(self):
        self.gas_labels = cfg.data.gas_labels
        self.regressors = self.load_regressors()
        self.compensator = joblib.load(Path(f"{cfg.paths.models_dir}/compensator.pkl"))
        self.classifier = joblib.load(
            Path(f"{cfg.paths.models_dir}/svm_classifier_compensated.pkl"),
        )

    def load_regressors(self) -> dict:
        models = {}
        for k, v in self.gas_labels.items():
            model = joblib.load(Path(f"{cfg.paths.models_dir}/regressor_{v}.pkl"))
            models[k] = model

        return models

    def predict(self, readings: np.ndarray) -> dict[str, str | float | bool]:
        aligned_readings = self.compensator.transform(readings.reshape(1, -1))

        gas_pred = self.classifier.predict(aligned_readings)
        confidence = np.max(self.classifier.predict_proba(aligned_readings))

        model = self.regressors[gas_pred[0]]

        concentration = model.predict(readings.reshape(1, -1))

        drift_risk = False

        return {
            "gas": self.gas_labels[gas_pred[0]],
            "confidence": confidence.item(),
            "concentration_ppm": concentration[0].item(),
            "drift_risk": drift_risk,
        }
