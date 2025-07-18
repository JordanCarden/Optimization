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

# --- Configuration ---
RESULTS_DIR = "experimental_results"
DATA_DIR = "data"
VARIANT = "AAV"
ALGORITHMS = sorted(["basin_hopping", "cmaes", "dual_annealing", "lshade", "pso"])

# Define standard scientific display names
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}

# Define a consistent color palette for all plots
CUSTOM_COLORS = ["#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"]
COLOR_MAP = dict(zip(ALGORITHMS, CUSTOM_COLORS))

# --- Helper Functions ---
def _load_experimental_trace() -> np.ndarray:
    """Load the experimental AAV trace."""
    mat_path = os.path.join(DATA_DIR, "experimental_data.mat")
    trace = loadmat(mat_path)[VARIANT].ravel().astype(float)
    return trace


def _load_best_sse_by_algorithm(num_points: int) -> pd.DataFrame:
    """Collect the best SSE from each optimization run."""
    results: List[Dict[str, float | str]] = []
    for algo in ALGORITHMS:
        display_name = DISPLAY_NAMES[algo]
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
            results.append({"algorithm": display_name, "rmse": rmse})
    return pd.DataFrame(results)


def _load_parameters(file_name: str) -> np.ndarray:
    """Load a parameter vector from DATA_DIR."""
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


# --- Plotting Functions ---
def plot_box_swarm() -> None:
    """Create a combined box and swarm plot of best run RMSE values."""
    exp_trace = _load_experimental_trace()
    df = _load_best_sse_by_algorithm(len(exp_trace))

    display_color_map = {DISPLAY_NAMES[algo]: color for algo, color in COLOR_MAP.items()}

    # Compute sorted algorithms by median RMSE (ascending)
    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    sorted_algorithms = medians.index.tolist()

    plt.figure(figsize=(10, 6))

    # Draw the gray box plot first
    sns.boxplot(
        data=df,
        x="algorithm",
        y="rmse",
        order=sorted_algorithms,
        color="lightgray",
        width=0.3,
        fliersize=0,
    )

    # Overlay the swarm plot with colored points
    sns.swarmplot(
        data=df,
        x="algorithm",
        y="rmse",
        hue="algorithm",
        palette=display_color_map,
        order=sorted_algorithms,
        size=5,
    )

    plt.xlabel("Optimizer")
    plt.ylabel("RMSE (AU)")
    plt.title("Distribution of Best RMSE Across 13 Runs on AAV Experimental Data")
    plt.xticks(rotation=45)
    plt.legend().remove()  # Remove legend since colors match x-axis
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "rmse_exp.png"), dpi=300)
    plt.close()


def plot_best_fits() -> None:
    """Compare best model fits from each optimizer to the data."""
    exp_trace = _load_experimental_trace()
    time_min = np.arange(0, 420 + 10, 10)

    rmse_values: Dict[str, float] = {}
    sims: Dict[str, np.ndarray] = {}

    for algo in ALGORITHMS:
        params = _load_parameters(f"{algo}_best_params.csv")
        sim = simulate_variant_response(
            params=params, model_params=MODEL_PARAMS, variant=VARIANT
        )
        display_name = DISPLAY_NAMES[algo]
        rmse = float(np.sqrt(np.mean((exp_trace - sim) ** 2)))
        rmse_values[display_name] = rmse
        sims[display_name] = sim

    # Sort by RMSE ascending
    sorted_items = sorted(rmse_values.items(), key=lambda x: x[1])
    sorted_names = [name for name, _ in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.plot(time_min, exp_trace, "o", color="black", label="Experimental")

    for name in sorted_names:
        sim = sims[name]
        algo = [k for k, v in DISPLAY_NAMES.items() if v == name][0]
        plt.plot(time_min, sim, label=name, color=COLOR_MAP[algo], linewidth=2.5)

    rmse_text = "RMSE\n" + "\n".join(
        f"{name}: {rmse_values[name]:.2f}" for name in sorted_names
    )
    plt.text(
        0.98,
        0.98,
        rmse_text,
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )
    plt.xlabel("Time (min)")
    plt.ylabel("GFP (AU)")
    plt.title("Best Model Fits vs Experimental Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "fit_exp.png"), dpi=300)
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_box_swarm()
    plot_best_fits()