from __future__ import annotations

import ast
import os
from typing import Sequence, Tuple

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


def _get_mase_df() -> Tuple[pd.DataFrame, list]:
    """Load and compute MASE for the dataframe."""
    df = pd.read_csv("synthetic_results.csv")
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]

    categories = sorted(df["category"].unique())
    ground_truth = {cat: _load_ground_truth(cat) for cat in categories}

    df["mase"] = df.apply(
        lambda row: _mase(_parse_params(row["params"]), ground_truth[row["category"]]),
        axis=1,
    )
    return df, categories


def plot_parameter_recovery() -> None:
    """Create a six-panel swarm plot figure of parameter recovery."""
    df, categories = _get_mase_df()

    # Compute sorted optimizers by overall median MASE (ascending)
    overall_medians = df.groupby("optimizer")["mase"].median().sort_values()
    sorted_optimizers = overall_medians.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()

    # Overall plot
    sns.boxplot(ax=ax_list[0], data=df, x="optimizer", y="mase", order=sorted_optimizers, color="lightgray", width=0.3, fliersize=0)
    sns.swarmplot(ax=ax_list[0], data=df, x="optimizer", y="mase", hue="optimizer",
                  palette=COLOR_MAP, order=sorted_optimizers, size=5)
    ax_list[0].set_title("Overall Parameter Recovery (MASE)")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("MASE")
    ax_list[0].set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
    ax_list[0].legend().remove()  # Remove legend since colors match x-axis
    for i, opt in enumerate(sorted_optimizers):
        median_val = overall_medians[opt]
        ax_list[0].text(i + 0.15, median_val, f'{median_val:.3f}', ha='left', va='center', fontsize=10, color='black')

    # Per-category plots
    for ax_idx, category in enumerate(categories, start=1):
        ax = ax_list[ax_idx]
        subset = df[df["category"] == category]
        subset_medians = subset.groupby("optimizer")["mase"].median()
        sns.boxplot(ax=ax, data=subset, x="optimizer", y="mase", order=sorted_optimizers, color="lightgray", width=0.3, fliersize=0)
        sns.swarmplot(ax=ax, data=subset, x="optimizer", y="mase", hue="optimizer",
                      palette=COLOR_MAP, order=sorted_optimizers, size=5)
        ax.set_title(DISPLAY_NAMES.get(category, category.replace("_", " ").title()))
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("MASE")
        ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
        ax.legend().remove()  # Remove legend
        for i, opt in enumerate(sorted_optimizers):
            median_val = subset_medians.get(opt, np.nan)
            if not np.isnan(median_val):
                ax.text(i + 0.15, median_val, f'{median_val:.3f}', ha='left', va='center', fontsize=10, color='black')

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/param_recovery.png", dpi=300)
    plt.show()


def plot_parameter_recovery_excluding_own() -> None:
    """Create a swarm plot of parameter recovery excluding own-dataset."""
    df, _ = _get_mase_df()

    # Exclude own-dataset performance
    df_filtered = df[df["optimizer"] != df["category"]]

    # Compute sorted optimizers by median MASE (ascending) on filtered data
    medians = df_filtered.groupby("optimizer")["mase"].median().sort_values()
    sorted_optimizers = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(ax=ax, data=df_filtered, x="optimizer", y="mase", order=sorted_optimizers, color="lightgray", width=0.3, fliersize=0)
    sns.swarmplot(ax=ax, data=df_filtered, x="optimizer", y="mase", hue="optimizer",
                  palette=COLOR_MAP, order=sorted_optimizers, size=5)
    ax.set_title("Parameter Recovery Excluding Own-Dataset (MASE)")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("MASE")
    ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
    ax.legend().remove()  # Remove legend since colors match x-axis
    ax.tick_params(axis="x", rotation=45)
    for i, opt in enumerate(sorted_optimizers):
        median_val = medians[opt]
        ax.text(i + 0.15, median_val, f'{median_val:.3f}', ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig("plots/param_recovery_exclusion.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_parameter_recovery()
    plot_parameter_recovery_excluding_own()