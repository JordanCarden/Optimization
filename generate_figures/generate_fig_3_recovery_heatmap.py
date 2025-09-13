#!/usr/bin/env python3
"""Generate Figure 3: Cross-Recovery Heatmap of Parameter Accuracy."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "python"))

ORDER = ["CMA-ES", "Basin Hopping", "L-SHADE", "Dual Annealing", "PSO"]
DISPLAY_MAP = {
    "cmaes": "CMA-ES",
    "basin_hopping": "Basin Hopping",
    "lshade": "L-SHADE",
    "dual_annealing": "Dual Annealing",
    "pso": "PSO",
}
COLORS = {
    "CMA-ES": "#E67E22",
    "Basin Hopping": "#F1C40F",
    "L-SHADE": "#9B59B6",
    "Dual Annealing": "#E91E63",
    "PSO": "#3A86FF",
}
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

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


def _parse_params(param_str: str) -> np.ndarray:
    """Parse a list-like parameter string into an array."""
    return np.asarray(ast.literal_eval(param_str), dtype=float)


def _load_ground_truth(category: str) -> np.ndarray:
    """Load ground-truth parameters for a dataset category."""
    df = pd.read_csv(DATA_DIR / f"{category}_best_params.csv")
    return df.iloc[0].to_numpy(dtype=float)


def _mase(found: Sequence[float], truth: Sequence[float]) -> float:
    """Compute the Mean Absolute Scaled Error (MASE)."""
    lb = np.asarray(LOWER_BOUNDS, dtype=float)
    ub = np.asarray(UPPER_BOUNDS, dtype=float)
    scale = ub - lb
    found_scaled = (np.asarray(found, dtype=float) - lb) / scale
    truth_scaled = (np.asarray(truth, dtype=float) - lb) / scale
    return float(np.mean(np.abs(found_scaled - truth_scaled)))


def main() -> None:
    """Create and save the cross-recovery heatmap."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
        }
    )

    df = pd.read_csv(ROOT / "synthetic_results.csv")
    df["generator"] = df["dataset"].str.rsplit("_", n=1).str[0]
    df["generator_display"] = df["generator"].map(DISPLAY_MAP)
    df["optimizer_display"] = df["optimizer"].map(DISPLAY_MAP)

    categories = df["generator"].unique()
    truth = {cat: _load_ground_truth(cat) for cat in categories}
    df["mase"] = df.apply(
        lambda r: _mase(_parse_params(r["params"]), truth[r["generator"]]), axis=1
    )

    pivot = df.pivot_table(
        index="generator_display",
        columns="optimizer_display",
        values="mase",
        aggfunc=np.median,
    )
    pivot = pivot.loc[ORDER, ORDER]

    fig, ax = plt.subplots(figsize=(7.09, 6.0))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Median MASE"},
        ax=ax,
    )
    ax.set_title("Parameter Recovery Performance (Median MASE)")
    ax.set_xlabel("Fitting Optimizer")
    ax.set_ylabel("Synthetic Data Generator")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_recovery_heatmap.pdf", bbox_inches="tight")
    fig.savefig(
        PLOTS_DIR / "fig3_recovery_heatmap.png", dpi=600, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
