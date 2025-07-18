"""Generate swarm plots for synthetic optimization results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_dataset_length() -> int:
    """Return the number of time points in a synthetic dataset."""
    sample_path = os.path.join("data", "basin_hopping_1.csv")
    df = pd.read_csv(sample_path)
    return len(df)


def plot_synthetic_swarm() -> None:
    """Plot overall and per-dataset optimizer performance."""
    df = pd.read_csv("synthetic_results.csv")
    num_points = _load_dataset_length()
    df["rmse"] = (df["min_sse"] / num_points) ** 0.5
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]

    categories = sorted(df["category"].unique())

    fig, axes = plt.subplots(3, 2, figsize=(12, 18), sharey=True)
    ax_list = axes.flatten()

    sns.swarmplot(ax=ax_list[0], data=df, x="optimizer", y="rmse")
    ax_list[0].set_title("Overall Optimizer Performance")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("RMSE")

    for ax, category in zip(ax_list[1:], categories):
        subset = df[df["category"] == category]
        sns.swarmplot(ax=ax, data=subset, x="optimizer", y="rmse")
        ax.set_title(f"{category.replace('_', ' ').title()}")
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("RMSE")

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_synthetic_swarm()
