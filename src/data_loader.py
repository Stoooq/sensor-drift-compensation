from pathlib import Path

import numpy as np
import polars as pl


def parse_data(path: Path) -> list[pl.DataFrame]:
    batches = []

    if not path.is_dir():
        raise ValueError("path is not a directory")

    for file_path in path.iterdir():
        if file_path.match("*.dat"):
            batch_id = int(file_path.stem[5:])
            all_features = []
            data_dict = {"label": [], "concentration": []}
            with file_path.open() as f:
                for line in f:
                    label, concentration, features = parse_line(line)
                    data_dict["label"].append(label)
                    data_dict["concentration"].append(concentration)
                    all_features.append(features)
            arr = np.array(all_features)
            data_dict = data_dict | {
                f"feature_{i}": arr[:, i - 1] for i in range(1, 129)
            }
            dataframe = pl.DataFrame(data_dict).with_columns(
                pl.lit(batch_id).alias("batch_id"),
            )
            batches.append(dataframe)

    return batches


def parse_line(line: str) -> tuple[int, float, list[float]]:
    parts = line.split()

    label, concentration = parts[0].split(";")
    features = [float(feature.split(":")[1]) for feature in parts[1:]]

    return int(label), float(concentration), features


def load_data(path: Path):
    batches = parse_data(path)

    data = pl.concat(batches)
    data = data.sort("batch_id")

    return data


def split_data(
    data: pl.DataFrame,
    split_after_batch: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_data = data.filter(pl.col("batch_id") <= split_after_batch)
    test_data = data.filter(pl.col("batch_id") > split_after_batch)

    return train_data, test_data
