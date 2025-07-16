"""Plot optimizer performance and best model fits."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from generate_data import MODEL_PARAMS
from simulate import simulate_variant_response

RESULTS_DIR = "experimental_results"
DATA_DIR = "data"
VARIANT = "AAV"
ALGORITHMS = ["basin_hopping", "dual_annealing", "lshade", "cmaes", "pso"]


def _load_experimental_trace() -> np.ndarray:
    """Load the experimental AAV trace."""
    mat_path = os.path.join(DATA_DIR, "experimental_data.mat")
    trace = loadmat(mat_path)[VARIANT].ravel().astype(float)
    return trace


def _load_best_sse_by_algorithm(num_points: int) -> pd.DataFrame:
    """Collect the best SSE from each optimization run.

    Args:
        num_points: Number of points in the experimental trace.

    Returns:
        DataFrame with columns ``algorithm`` and ``rmse``.
    """
    results: List[Dict[str, float]] = []

    for algo in ALGORITHMS:
        for run_idx in range(1, 14):
            file_name = f"{algo}_history_variant_{VARIANT}_run_{run_idx}.csv"
            path = os.path.join(RESULTS_DIR, file_name)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if "sse" not in df:
                raise KeyError(f"sse column not found in {path}")

            best_sse = df["sse"].min()
            rmse = np.sqrt(best_sse / num_points)
            results.append({"algorithm": algo, "rmse": rmse})

    return pd.DataFrame(results)


def plot_swarm() -> None:
    """Create a swarm plot of best run RMSE values."""
    exp_trace = _load_experimental_trace()
    df = _load_best_sse_by_algorithm(len(exp_trace))

    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x="algorithm", y="rmse")
    plt.xlabel("Optimizer")
    plt.ylabel("RMSE")
    plt.title("Best RMSE per Run")
    plt.tight_layout()
    plt.show()


def _load_parameters(file_name: str) -> np.ndarray:
    """Load a parameter vector from ``DATA_DIR``."""
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def plot_best_fits() -> None:
    """Compare best model fits from each optimizer to the data."""
    exp_trace = _load_experimental_trace()
    time_min = np.arange(0, 420 + 10, 10)

    rmse_values: Dict[str, float] = {}
    plt.figure(figsize=(10, 6))
    plt.plot(time_min, exp_trace, "o", label="Experimental")

    for algo in ALGORITHMS:
        params = _load_parameters(f"{algo}_best_params.csv")
        sim = simulate_variant_response(
            params=params, model_params=MODEL_PARAMS, variant=VARIANT
        )
        plt.plot(time_min, sim, label=algo)
        rmse = float(np.sqrt(np.mean((exp_trace - sim) ** 2)))
        rmse_values[algo] = rmse

    rmse_text = "\n".join(f"{k}: {v:.2f}" for k, v in rmse_values.items())
    plt.text(
        0.02,
        0.98,
        rmse_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )
    plt.xlabel("Time (min)")
    plt.ylabel("GFP (AU)")
    plt.title("Best Model Fits vs Experimental Data")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_swarm()
    plot_best_fits()
