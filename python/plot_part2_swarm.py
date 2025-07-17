"""Generate swarm plots for synthetic optimizer results."""

from __future__ import annotations

import ast
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_part1 import ALGORITHMS, CUSTOM_COLORS, DISPLAY_NAMES

# Normalization bounds copied from ``fit_synthetic.py``
LOWER_BOUNDS: np.ndarray = np.array([
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
])
UPPER_BOUNDS: np.ndarray = np.array([
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
])


def _load_gt_params(source: str) -> np.ndarray:
    """Load ground truth parameters for a given data source."""
    path = os.path.join("data", f"{source}_best_params.csv")
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def _compute_mase(best: Sequence[float], gt: Sequence[float]) -> float:
    """Compute Mean Absolute Scaled Error between two parameter vectors."""
    best = np.asarray(best, dtype=float)
    gt = np.asarray(gt, dtype=float)
    normalized_best = (best - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
    normalized_gt = (gt - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
    return float(np.mean(np.abs(normalized_best - normalized_gt)))


def plot_rmse_swarm(df: pd.DataFrame) -> None:
    """Create a swarm plot of best RMSE values."""
    display_order = [DISPLAY_NAMES[a] for a in ALGORITHMS]
    palette = dict(zip(display_order, CUSTOM_COLORS))

    plt.figure(figsize=(8, 6))
    sns.swarmplot(
        data=df,
        x="algorithm",
        y="rmse",
        order=display_order,
        palette=palette,
        size=5,
    )
    plt.xlabel("Optimizer")
    plt.ylabel("RMSE (AU)")
    plt.title("Distribution of Best RMSE Across 150 Runs on Synthetic Data")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "synthetic_rmse_swarm.png"), dpi=300)
    plt.close()


def plot_mase_swarm(df: pd.DataFrame) -> None:
    """Create a swarm plot of MASE values for parameter recovery."""
    display_order = [DISPLAY_NAMES[a] for a in ALGORITHMS]
    palette = dict(zip(display_order, CUSTOM_COLORS))

    plt.figure(figsize=(8, 6))
    sns.swarmplot(
        data=df,
        x="algorithm",
        y="mase",
        order=display_order,
        palette=palette,
        size=5,
    )
    plt.xlabel("Optimizer")
    plt.ylabel("MASE")
    plt.title(
        "Distribution of MASE for Parameter Recovery Across 150 Runs on Synthetic Data"
    )
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "synthetic_mase_swarm.png"), dpi=300)
    plt.close()


def main() -> None:
    """Entry point for generating the Part 2 swarm plots."""
    df = pd.read_csv("synthetic_results.csv")
    df = df.rename(
        columns={"fitting_algo": "algorithm", "best_rmse": "rmse"}
    )

    df["algorithm"] = df["algorithm"].map(DISPLAY_NAMES)

    # Compute MASE for each row
    mase_values = []
    for _, row in df.iterrows():
        best_params = ast.literal_eval(row["best_params"])
        gt_params = _load_gt_params(row["gt_source"])
        mase_values.append(_compute_mase(best_params, gt_params))
    df["mase"] = mase_values

    os.makedirs("plots", exist_ok=True)
    plot_rmse_swarm(df)
    plot_mase_swarm(df)


if __name__ == "__main__":
    main()
