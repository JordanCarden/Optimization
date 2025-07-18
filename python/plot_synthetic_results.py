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

    # Define display names
    DISPLAY_NAMES = {
        "basin_hopping": "Basin Hopping",
        "cmaes": "CMA-ES",
        "dual_annealing": "Dual Annealing",
        "lshade": "L-SHADE",
        "pso": "PSO",
    }

    # Define a consistent color palette
    ALGORITHMS = list(DISPLAY_NAMES.keys())
    CUSTOM_COLORS = ["#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"]
    COLOR_MAP = dict(zip(ALGORITHMS, CUSTOM_COLORS))

    # Compute sorted optimizers by overall median RMSE (ascending)
    medians = df.groupby("optimizer")["rmse"].median().sort_values()
    sorted_optimizers = medians.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()

    # Overall plot
    sns.boxplot(ax=ax_list[0], data=df, x="optimizer", y="rmse", order=sorted_optimizers, color="lightgray", width=0.3, fliersize=0)
    sns.swarmplot(ax=ax_list[0], data=df, x="optimizer", y="rmse", hue="optimizer",
                  palette=COLOR_MAP, order=sorted_optimizers, size=5)
    ax_list[0].set_title("Overall Optimizer Performance")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("RMSE")
    ax_list[0].set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
    ax_list[0].legend().remove()  # Remove legend since colors match x-axis

    # Per-category plots
    for ax, category in zip(ax_list[1:], categories):
        subset = df[df["category"] == category]
        sns.boxplot(ax=ax, data=subset, x="optimizer", y="rmse", order=sorted_optimizers, color="lightgray", width=0.3, fliersize=0)
        sns.swarmplot(ax=ax, data=subset, x="optimizer", y="rmse", hue="optimizer",
                      palette=COLOR_MAP, order=sorted_optimizers, size=5)
        ax.set_title(DISPLAY_NAMES.get(category, category.replace("_", " ").title()))
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("RMSE")
        ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
        ax.legend().remove()  # Remove legend

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/rmse_syn.png")
    plt.show()


if __name__ == "__main__":
    plot_synthetic_swarm()