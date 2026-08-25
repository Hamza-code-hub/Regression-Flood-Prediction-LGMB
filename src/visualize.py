"""
Visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_path(path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    path,
):
    path = _prepare_path(
        path
    )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    minimum = min(
        y_true.min(),
        y_pred.min(),
    )

    maximum = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.55,
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual"
    )

    plt.ylabel(
        "Predicted"
    )

    plt.title(
        "Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_residuals(
    y_true,
    y_pred,
    path,
):
    path = _prepare_path(
        path
    )

    residuals = (
        np.asarray(y_true)
        - np.asarray(y_pred)
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.55,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_target_distribution(
    y,
    path,
):
    path = _prepare_path(
        path
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        y,
        bins=30,
        edgecolor="black",
    )

    plt.xlabel(
        "Target"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Target Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_feature_importance(
    pipeline,
    path_csv,
    path_plot,
):
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = model.feature_importances_

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    path_csv = _prepare_path(
        path_csv
    )

    frame.to_csv(
        path_csv,
        index=False,
    )

    plot_frame = frame.head(
        20
    ).sort_values(
        "importance"
    )

    path_plot = _prepare_path(
        path_plot
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        plot_frame["feature"],
        plot_frame["importance"],
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "LightGBM Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        path_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return frame
