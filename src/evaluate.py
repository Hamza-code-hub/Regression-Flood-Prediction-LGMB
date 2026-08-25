"""
Regression evaluation utilities.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .config import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET,
)

from .data import load_csv

from .utils import (
    safe_mape,
    save_json,
)

from .visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
)


def regression_metrics(
    y_true,
    y_pred,
):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(rmse),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "mape_percent": safe_mape(
            y_true,
            y_pred,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained "
            "LightGBM flood pipeline"
        )
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
    )

    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = load_csv(
        args.data
    )

    if (
        args.target
        not in dataframe.columns
    ):
        raise ValueError(
            f"Target '{args.target}' "
            "not found."
        )

    y_true = dataframe[
        args.target
    ]

    drop_columns = [
        args.target
    ]

    if (
        args.id_column
        in dataframe.columns
    ):
        drop_columns.append(
            args.id_column
        )

    X = dataframe.drop(
        columns=drop_columns
    )

    pipeline = joblib.load(
        args.model
    )

    y_pred = pipeline.predict(
        X
    )

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        metrics,
        output_dir
        / "metrics.json",
    )

    predictions = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": (
                np.asarray(
                    y_true
                )
                - y_pred
            ),
        }
    )

    predictions.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        y_true,
        y_pred,
        output_dir
        / "actual_vs_predicted.png",
    )

    plot_residuals(
        y_true,
        y_pred,
        output_dir
        / "residuals.png",
    )

    print(metrics)


if __name__ == "__main__":
    main()
