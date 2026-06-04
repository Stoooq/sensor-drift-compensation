from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    data_dir: str
    results_dir: str
    models_dir: str


class DataConfig(BaseModel):
    n_sensors: int
    n_features_per_sensor: int
    n_classes: int
    train_batches: list[int]
    test_batches: list[int]
    gas_labels: dict[int, str]


class FeaturesConfig(BaseModel):
    pca_n_components: int


class ModelsConfig(BaseModel):
    classifier_types: list[str]
    default_classifier: str


class AppConfig(BaseModel):
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    with Path(path).open() as f:
        return AppConfig(**yaml.safe_load(f))
