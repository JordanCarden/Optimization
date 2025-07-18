"""Visualize optimizer parameter recovery accuracy using MASE."""

from __future__ import annotations

import ast
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Parameter bounds copied from ``fit_synthetic.py``
LOWER_BOUNDS: Sequence[float] = [
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0005,
    0.0001,
    0.0001,
    0.00005,
    0.0001,
    0.0001,
    0.00005,
    100,
    0.00000001,
    100,
    100,
    100,
    0.00000001,
    0.00000001,
    0.5,
    0.0001,
    100,
]

UPPER_BOUNDS: Sequence[float] = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    1000000,
    0.00001,
    1000000,
    1000000,
    1000000,
    0.00001,
    0.00001,
    5,
    0.01,
    1000000,
]


def _load_ground_truth(category: str) -> np.ndarray:
    """Return ground truth parameter vector for a dataset category."""
    path = os.path.join("data", f"{category}_best_params.csv")
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def _parse_params(param_str: str) -> np.ndarray:
    """Parse a list-like string of parameters."""
    return np.asarray(ast.literal_eval(param_str), dtype=float)


def _mase(found: Sequence[float], truth: Sequence[float]) -> float:
    """Compute the Mean Absolute Scaled Error (MASE)."""
    lb = np.asarray(LOWER_BOUNDS, dtype=float)
    ub = np.asarray(UPPER_BOUNDS, dtype=float)
    scale = ub - lb
    found_scaled = (np.asarray(found, dtype=float) - lb) / scale
    truth_scaled = (np.asarray(truth, dtype=float) - lb) / scale
    return float(np.mean(np.abs(found_scaled - truth_scaled)))


def plot_parameter_recovery() -> None:
    """Create a six-panel swarm plot figure of parameter recovery."""
    df = pd.read_csv("synthetic_results.csv")
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]

    # Preload ground truth parameters for all categories
    categories = sorted(df["category"].unique())
    ground_truth = {cat: _load_ground_truth(cat) for cat in categories}

    df["mase"] = df.apply(
        lambda row: _mase(_parse_params(row["params"]), ground_truth[row["category"]]),
        axis=1,
    )

    display_names = {
        "basin_hopping": "Basin Hopping",
        "cmaes": "CMA-ES",
        "dual_annealing": "Dual Annealing",
        "lshade": "L-SHADE",
        "pso": "PSO",
    }
    algorithms = list(display_names.keys())
    colors = ["#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"]
    color_map = dict(zip(algorithms, colors))

    medians = df.groupby("optimizer")["mase"].median().sort_values()
    sorted_opts = medians.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()

    sns.swarmplot(
        ax=ax_list[0],
        data=df,
        x="optimizer",
        y="mase",
        order=sorted_opts,
        palette=color_map,
    )
    ax_list[0].set_title("Overall Parameter Recovery (MASE)")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("MASE")
    ax_list[0].set_xticklabels([display_names.get(opt, opt) for opt in sorted_opts])

    for ax, category in zip(ax_list[1:], categories):
        subset = df[df["category"] == category]
        sns.swarmplot(
            ax=ax,
            data=subset,
            x="optimizer",
            y="mase",
            order=sorted_opts,
            palette=color_map,
        )
        title = (
            "Parameter Recovery for "
            f"{display_names.get(category, category.title())} Ground Truth"
        )
        ax.set_title(title)
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("MASE")
        ax.set_xticklabels([display_names.get(opt, opt) for opt in sorted_opts])

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/parameter_recovery_mase.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_parameter_recovery()
