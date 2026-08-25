"""
Dataset loading and validation.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_SIZE


def load_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return dataframe


def prepare_training_data(
    dataframe,
    target_column,
    id_column="id",
    test_size=TEST_SIZE,
    random_seed=RANDOM_SEED,
):
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    y = dataframe[target_column].copy()

    drop_columns = [target_column]

    if (
        id_column
        and id_column in dataframe.columns
    ):
        drop_columns.append(id_column)

    X = dataframe.drop(
        columns=drop_columns
    ).copy()

    if X.empty:
        raise ValueError(
            "No feature columns remain after "
            "removing the target and ID."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
    )


def prepare_inference_data(
    dataframe,
    id_column="id",
):
    ids = None

    if (
        id_column
        and id_column in dataframe.columns
    ):
        ids = dataframe[
            id_column
        ].copy()

        dataframe = dataframe.drop(
            columns=[id_column]
        )

    return dataframe, ids
