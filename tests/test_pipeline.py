import numpy as np
import pandas as pd

from src.data import (
    prepare_training_data,
)

from src.model import build_pipeline
from src.utils import safe_mape


def test_safe_mape_is_finite():
    y_true = np.array(
        [0.0, 1.0, 2.0]
    )

    y_pred = np.array(
        [0.1, 1.1, 1.9]
    )

    value = safe_mape(
        y_true,
        y_pred,
    )

    assert np.isfinite(value)


def test_training_pipeline():
    dataframe = pd.DataFrame(
        {
            "id": range(20),
            "rainfall": np.linspace(
                10,
                100,
                20,
            ),
            "soil": [
                "dry",
                "wet",
            ] * 10,
            "FloodProbability":
                np.linspace(
                    0.1,
                    0.9,
                    20,
                ),
        }
    )

    X_train, X_valid, \
    y_train, y_valid = (
        prepare_training_data(
            dataframe,
            target_column=(
                "FloodProbability"
            ),
            id_column="id",
            test_size=0.25,
        )
    )

    pipeline = build_pipeline(
        X_train,
        n_estimators=10,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    prediction = pipeline.predict(
        X_valid
    )

    assert len(prediction) == len(
        y_valid
    )
