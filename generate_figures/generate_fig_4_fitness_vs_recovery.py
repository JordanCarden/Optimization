#!/usr/bin/env python3
"""Generate Figure 4: Fitness vs. Parameter Recovery Scatter Plot."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import spearmanr

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
    """Create and save the fitness vs recovery scatter plot."""
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
    df["optimizer_display"] = df["optimizer"].map(DISPLAY_MAP)

    categories = df["generator"].unique()
    truth = {cat: _load_ground_truth(cat) for cat in categories}
    df["mase"] = df.apply(
        lambda r: _mase(_parse_params(r["params"]), truth[r["generator"]]), axis=1
    )

    num_points = loadmat(DATA_DIR / "experimental_data.mat")["AAV"].size
    df["rmse"] = np.sqrt(df["min_sse"] / num_points)

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    for name in ORDER:
        subset = df[df["optimizer_display"] == name]
        ax.scatter(
            subset["rmse"],
            subset["mase"],
            color=COLORS[name],
            s=16,
            label=name,
        )
        sns.regplot(
            data=subset,
            x="rmse",
            y="mase",
            scatter=False,
            color=COLORS[name],
            ax=ax,
            ci=None,
        )

    rho, pval = spearmanr(df["rmse"], df["mase"])
    ax.text(
        0.05,
        0.95,
        f"Spearman \u03c1={rho:.2f}\n" f"p={pval:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.set_title("Fit Quality vs. Parameter Recovery")
    ax.set_xlabel("Final RMSE")
    ax.set_ylabel("Parameter MASE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_fitness_vs_recovery.pdf", bbox_inches="tight")
    fig.savefig(
        PLOTS_DIR / "fig4_fitness_vs_recovery.png", dpi=600, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
