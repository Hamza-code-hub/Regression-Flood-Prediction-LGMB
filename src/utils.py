"""
Shared project utilities.
"""

import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def save_json(data, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def safe_mape(
    y_true,
    y_pred,
    epsilon=1e-8,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = np.maximum(
        np.abs(y_true),
        epsilon,
    )

    return float(
        np.mean(
            np.abs(
                (
                    y_true
                    - y_pred
                )
                / denominator
            )
        )
        * 100.0
    )
