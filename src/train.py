"""
Train the LightGBM flood-regression pipeline.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET,
    MODEL_DIR,
    MODEL_FILENAME,
    OUTPUT_DIR,
    RANDOM_SEED,
)

from .data import (
    load_csv,
    prepare_training_data,
)

from .evaluate import regression_metrics
from .model import build_pipeline

from .utils import (
    save_json,
    set_seed,
)

from .visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_target_distribution,
    save_feature_importance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM "
            "for flood regression"
        )
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
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(
        RANDOM_SEED
    )

    dataframe = load_csv(
        args.data
    )

    X_train, X_valid, \
    y_train, y_valid = (
        prepare_training_data(
            dataframe,
            target_column=args.target,
            id_column=args.id_column,
            test_size=args.test_size,
        )
    )

    pipeline = build_pipeline(
        X_train,
        n_estimators=(
            args.estimators
        ),
        learning_rate=(
            args.learning_rate
        ),
        num_leaves=(
            args.num_leaves
        ),
        max_depth=(
            args.max_depth
        ),
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_valid
    )

    metrics = regression_metrics(
        y_valid,
        predictions,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / MODEL_FILENAME
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    save_json(
        metrics,
        OUTPUT_DIR
        / "metrics.json",
    )

    validation_frame = pd.DataFrame(
        {
            "actual": (
                y_valid
                .reset_index(
                    drop=True
                )
            ),
            "predicted": predictions,
        }
    )

    validation_frame[
        "residual"
    ] = (
        validation_frame[
            "actual"
        ]
        - validation_frame[
            "predicted"
        ]
    )

    validation_frame.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        y_valid,
        predictions,
        OUTPUT_DIR
        / "actual_vs_predicted.png",
    )

    plot_residuals(
        y_valid,
        predictions,
        OUTPUT_DIR
        / "residuals.png",
    )

    plot_target_distribution(
        dataframe[
            args.target
        ],
        OUTPUT_DIR
        / "target_distribution.png",
    )

    save_feature_importance(
        pipeline,
        OUTPUT_DIR
        / "feature_importance.csv",
        OUTPUT_DIR
        / "feature_importance.png",
    )

    print(
        "\nTraining complete"
    )

    print(
        f"Model: {model_path}"
    )

    for key, value in (
        metrics.items()
    ):
        print(
            f"{key}: "
            f"{value:.6f}"
        )


if __name__ == "__main__":
    main()
