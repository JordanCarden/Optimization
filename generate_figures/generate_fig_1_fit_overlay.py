#!/usr/bin/env python3
"""Generate Figure 1: Model Fits and Variability on Experimental Data.

This script overlays experimental GFP measurements with simulated traces from
multiple optimizers. The median and interquartile range (IQR) of the simulated
traces are shown to highlight run-to-run variability.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Ensure the `python` directory is on the Python path for importing helpers.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "python"))

from generate_data import MODEL_PARAMS  # noqa: E402
from simulate import simulate_variant_response  # noqa: E402

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


def _load_experimental_trace() -> np.ndarray:
    """Load the experimental GFP time course for the AAV variant."""
    mat = loadmat(DATA_DIR / "experimental_data.mat")
    return mat[VARIANT].ravel().astype(float)


def _simulate_runs(optimizer: str) -> np.ndarray:
    """Return simulated traces for all runs of an optimizer.

    Args:
        optimizer: Internal optimizer name.

    Returns:
        Array of shape (runs, time) with simulated GFP traces.
    """
    traces: List[np.ndarray] = []
    for run_idx in range(1, 14):
        file_name = f"{optimizer}_history_variant_{VARIANT}_run_{run_idx}.csv"
        path = RESULTS_DIR / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        best_row = df.loc[df["sse"].idxmin()]
        params = np.asarray(ast.literal_eval(best_row["params"]), dtype=float)
        sim = simulate_variant_response(
            params=params, model_params=MODEL_PARAMS, variant=VARIANT
        )
        traces.append(sim)
    return np.asarray(traces)


def main() -> None:
    """Create and save the fit overlay figure."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
        }
    )

    exp_trace = _load_experimental_trace()
    time_min = np.arange(0, exp_trace.size * 10, 10)

    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    ax.plot(
        time_min,
        exp_trace,
        "o",
        color="black",
        markersize=4,
        label="Experimental",
    )

    for internal in INTERNAL_NAMES:
        display = DISPLAY_MAP[internal]
        traces = _simulate_runs(internal)
        if traces.size == 0:
            continue
        median = np.quantile(traces, 0.5, axis=0)
        q1 = np.quantile(traces, 0.25, axis=0)
        q3 = np.quantile(traces, 0.75, axis=0)
        ax.plot(time_min, median, color=COLORS[display], linewidth=1.0, label=display)
        ax.fill_between(
            time_min,
            q1,
            q3,
            color=COLORS[display],
            alpha=0.2,
        )

    ax.set_title("Model Fits and Variability on Experimental Data")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("GFP (AU)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig1_fit_overlay.pdf", bbox_inches="tight")
    fig.savefig(
        PLOTS_DIR / "fig1_fit_overlay.png", dpi=600, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
