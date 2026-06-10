from pathlib import Path

import polars as pl

from src.config import load_config
from src.data_loader import load_data, split_data

cfg = load_config(path=Path("config.yaml"))


def test_data_loader():
    data = load_data(Path(cfg.paths.data_dir))
    train_data = data.select(
        pl.exclude("label", "concentration", "batch_id"),
    ).to_numpy()

    assert train_data.shape[1] == 128


def test_data_split():
    data = load_data(Path(cfg.paths.data_dir))
    train_data, test_data = split_data(data, split_after_batch=6)

    assert train_data["batch_id"].max() <= 6
    assert test_data["batch_id"].min() >= 7
