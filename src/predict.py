"""
Apply a trained flood-regression pipeline
to unseen CSV data.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_ID_COLUMN

from .data import (
    load_csv,
    prepare_inference_data,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/"
            "test_predictions.csv"
        ),
    )

    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
    )

    parser.add_argument(
        "--prediction-column",
        default="prediction",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = load_csv(
        args.input
    )

    features, ids = (
        prepare_inference_data(
            dataframe,
            args.id_column,
        )
    )

    pipeline = joblib.load(
        args.model
    )

    predictions = pipeline.predict(
        features
    )

    output = pd.DataFrame()

    if ids is not None:
        output[
            args.id_column
        ] = ids.values

    output[
        args.prediction_column
    ] = predictions

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Predictions saved to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
