#!/usr/bin/env python3
"""Generate Figure 2: Convergence and Efficiency of Optimizers.

The figure contains two panels showing convergence behaviour on experimental
runs and the distribution of function evaluations required to reach a common
RMSE threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "python"))

ORDER = ["CMA-ES", "Basin Hopping", "L-SHADE", "Dual Annealing", "PSO"]
INTERNAL_NAMES = ["cmaes", "basin_hopping", "lshade", "dual_annealing", "pso"]
DISPLAY_MAP = dict(zip(INTERNAL_NAMES, ORDER))
COLORS = {
    "CMA-ES": "#E67E22",
    "Basin Hopping": "#F1C40F",
    "L-SHADE": "#9B59B6",
    "Dual Annealing": "#E91E63",
    "PSO": "#3A86FF",
}
RESULTS_DIR = ROOT / "experimental_results"
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
VARIANT = "AAV"
MAX_EVALS = 5000


def _load_experimental_trace() -> np.ndarray:
    """Return the experimental GFP time course to determine trace length."""
    mat = loadmat(DATA_DIR / "experimental_data.mat")
    return mat[VARIANT].ravel().astype(float)


def _load_histories() -> Dict[str, List[np.ndarray]]:
    """Load RMSE histories for each optimizer.

    Returns:
        Mapping of display name to a list of RMSE arrays, each of length
        ``MAX_EVALS`` with trailing ``NaN`` values if a run terminated early.
    """
    num_points = _load_experimental_trace().size
    histories: Dict[str, List[np.ndarray]] = {name: [] for name in ORDER}
    for internal in INTERNAL_NAMES:
        display = DISPLAY_MAP[internal]
        for run_idx in range(1, 14):
            file_name = f"{internal}_history_variant_{VARIANT}_run_{run_idx}.csv"
            path = RESULTS_DIR / file_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            sse = df["sse"].to_numpy(dtype=float)
            rmse = np.sqrt(sse / num_points)
            arr = np.full(MAX_EVALS, np.nan)
            arr[: min(MAX_EVALS, rmse.size)] = rmse[:MAX_EVALS]
            histories[display].append(arr)
    return histories


def _compute_threshold(histories: Dict[str, List[np.ndarray]]) -> float:
    """Compute the median final RMSE across all runs."""
    finals: List[float] = []
    for runs in histories.values():
        for arr in runs:
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                finals.append(float(valid[-1]))
    return float(np.median(finals))


def main() -> None:
    """Create and save the convergence and efficiency figure."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
        }
    )

    histories = _load_histories()
    threshold = _compute_threshold(histories)
    evals = np.arange(1, MAX_EVALS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.5))

    # Panel A: Convergence curves.
    ax = axes[0]
    for display in ORDER:
        runs = histories.get(display, [])
        if not runs:
            continue
        data = np.vstack(runs)
        median = np.nanmedian(data, axis=0)
        q1 = np.nanpercentile(data, 25, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        ax.plot(evals, median, color=COLORS[display], linewidth=1.0, label=display)
        ax.fill_between(evals, q1, q3, color=COLORS[display], alpha=0.2)
    ax.set_title("(A) Convergence on Experimental Data")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("RMSE (AU)")
    ax.legend()

    # Panel B: Evaluations to reach threshold RMSE.
    records: List[Dict[str, float]] = []
    for display, runs in histories.items():
        for arr in runs:
            indices = np.where(arr <= threshold)[0]
            if indices.size > 0:
                records.append({"optimizer": display, "evals": indices[0] + 1})
    df = pd.DataFrame(records)
    if not df.empty:
        order = (
            df.groupby("optimizer")["evals"].median().sort_values().index.tolist()
        )
    else:
        order = ORDER
    ax = axes[1]
    sns.boxplot(
        data=df,
        x="optimizer",
        y="evals",
        order=order,
        palette=[COLORS[o] for o in order],
        ax=ax,
    )
    ax.set_title("(B) Evaluations to Reach Threshold RMSE")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Function Evaluations")

    fig.suptitle("Optimizer Convergence and Efficiency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig2_convergence.pdf", bbox_inches="tight")
    fig.savefig(
        PLOTS_DIR / "fig2_convergence.png", dpi=600, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
