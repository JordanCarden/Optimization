from __future__ import annotations

import ast
import os
from typing import Dict, List, Sequence, Tuple

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

# Display names and color mapping shared across plots.
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}
ALGORITHMS = list(DISPLAY_NAMES.keys())
CUSTOM_COLORS = ["#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"]
COLOR_MAP = dict(zip(ALGORITHMS, CUSTOM_COLORS))


# ---------------------------------------------------------------------------
# Experimental results plotting
# ---------------------------------------------------------------------------

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
    """Load a parameter vector from ``DATA_DIR``."""
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def plot_box_swarm() -> None:
    """Create a combined box and swarm plot of best run RMSE values."""
    exp_trace = _load_experimental_trace()
    df = _load_best_sse_by_algorithm(len(exp_trace))

    display_color_map = {DISPLAY_NAMES[a]: COLOR_MAP[a] for a in ALGORITHMS}

    # Compute sorted algorithms by median RMSE (ascending).
    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    sorted_algorithms = medians.index.tolist()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="algorithm",
        y="rmse",
        order=sorted_algorithms,
        color="lightgray",
        width=0.3,
        fliersize=0,
    )
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
    plt.legend().remove()
    for i, algo in enumerate(sorted_algorithms):
        median_val = medians[algo]
        plt.text(
            i + 0.15,
            median_val,
            f"{median_val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )
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
            params=params,
            model_params=MODEL_PARAMS,
            variant=VARIANT,
        )
        display_name = DISPLAY_NAMES[algo]
        rmse = float(np.sqrt(np.mean((exp_trace - sim) ** 2)))
        rmse_values[display_name] = rmse
        sims[display_name] = sim

    sorted_items = sorted(rmse_values.items(), key=lambda x: x[1])
    sorted_names = [name for name, _ in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.plot(time_min, exp_trace, "o", color="black", label="Experimental")
    for name in sorted_names:
        sim = sims[name]
        algo = [k for k, v in DISPLAY_NAMES.items() if v == name][0]
        plt.plot(time_min, sim, label=name, color=COLOR_MAP[algo], linewidth=2.5)

    rmse_text = "RMSE\n" + "\n".join(f"{n}: {rmse_values[n]:.2f}" for n in sorted_names)
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


# ---------------------------------------------------------------------------
# Parameter recovery plotting
# ---------------------------------------------------------------------------

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
    overall_medians = df.groupby("optimizer")["mase"].median().sort_values()
    overall_sorted = overall_medians.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()

    sns.boxplot(
        ax=ax_list[0],
        data=df,
        x="optimizer",
        y="mase",
        order=overall_sorted,
        color="lightgray",
        width=0.3,
        fliersize=0,
    )
    sns.swarmplot(
        ax=ax_list[0],
        data=df,
        x="optimizer",
        y="mase",
        hue="optimizer",
        palette=COLOR_MAP,
        order=overall_sorted,
        size=5,
    )
    ax_list[0].set_title("Overall Parameter Recovery (MASE)")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("MASE")
    ax_list[0].set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in overall_sorted])
    ax_list[0].legend().remove()
    for i, opt in enumerate(overall_sorted):
        median_val = overall_medians[opt]
        ax_list[0].text(
            i + 0.15,
            median_val,
            f"{median_val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )

    for ax_idx, category in enumerate(categories, start=1):
        ax = ax_list[ax_idx]
        subset = df[df["category"] == category]
        subset_medians = (
            subset.groupby("optimizer")["mase"].median().sort_values()
        )
        subset_sorted = subset_medians.index.tolist()
        sns.boxplot(
            ax=ax,
            data=subset,
            x="optimizer",
            y="mase",
            order=subset_sorted,
            color="lightgray",
            width=0.3,
            fliersize=0,
        )
        sns.swarmplot(
            ax=ax,
            data=subset,
            x="optimizer",
            y="mase",
            hue="optimizer",
            palette=COLOR_MAP,
            order=subset_sorted,
            size=5,
        )
        ax.set_title(DISPLAY_NAMES.get(category, category.replace("_", " ").title()))
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("MASE")
        ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in subset_sorted])
        ax.legend().remove()
        for i, opt in enumerate(subset_sorted):
            median_val = subset_medians.get(opt, np.nan)
            if not np.isnan(median_val):
                ax.text(
                    i + 0.15,
                    median_val,
                    f"{median_val:.3f}",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/param_recovery.png", dpi=300)
    plt.show()


def plot_parameter_recovery_excluding_own() -> None:
    """Create a swarm plot of parameter recovery excluding own-dataset."""
    df, _ = _get_mase_df()

    df_filtered = df[df["optimizer"] != df["category"]]
    medians = df_filtered.groupby("optimizer")["mase"].median().sort_values()
    sorted_optimizers = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        ax=ax,
        data=df_filtered,
        x="optimizer",
        y="mase",
        order=sorted_optimizers,
        color="lightgray",
        width=0.3,
        fliersize=0,
    )
    sns.swarmplot(
        ax=ax,
        data=df_filtered,
        x="optimizer",
        y="mase",
        hue="optimizer",
        palette=COLOR_MAP,
        order=sorted_optimizers,
        size=5,
    )
    ax.set_title("Parameter Recovery Excluding Own-Dataset (MASE)")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("MASE")
    ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in sorted_optimizers])
    ax.legend().remove()
    ax.tick_params(axis="x", rotation=45)
    for i, opt in enumerate(sorted_optimizers):
        median_val = medians[opt]
        ax.text(
            i + 0.15,
            median_val,
            f"{median_val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()
    plt.savefig("plots/param_recovery_exclusion.png", dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Synthetic dataset results plotting
# ---------------------------------------------------------------------------

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
    overall_medians = df.groupby("optimizer")["rmse"].median().sort_values()
    overall_sorted = overall_medians.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()

    sns.boxplot(
        ax=ax_list[0],
        data=df,
        x="optimizer",
        y="rmse",
        order=overall_sorted,
        color="lightgray",
        width=0.3,
        fliersize=0,
    )
    sns.swarmplot(
        ax=ax_list[0],
        data=df,
        x="optimizer",
        y="rmse",
        hue="optimizer",
        palette=COLOR_MAP,
        order=overall_sorted,
        size=5,
    )
    ax_list[0].set_title("Overall Optimizer Performance")
    ax_list[0].set_xlabel("Optimizer")
    ax_list[0].set_ylabel("RMSE")
    ax_list[0].set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in overall_sorted])
    ax_list[0].legend().remove()
    for i, opt in enumerate(overall_sorted):
        median_val = overall_medians[opt]
        ax_list[0].text(
            i + 0.15,
            median_val,
            f"{median_val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )

    for ax_idx, category in enumerate(categories, start=1):
        ax = ax_list[ax_idx]
        subset = df[df["category"] == category]
        subset_medians = (
            subset.groupby("optimizer")["rmse"].median().sort_values()
        )
        subset_sorted = subset_medians.index.tolist()
        sns.boxplot(
            ax=ax,
            data=subset,
            x="optimizer",
            y="rmse",
            order=subset_sorted,
            color="lightgray",
            width=0.3,
            fliersize=0,
        )
        sns.swarmplot(
            ax=ax,
            data=subset,
            x="optimizer",
            y="rmse",
            hue="optimizer",
            palette=COLOR_MAP,
            order=subset_sorted,
            size=5,
        )
        ax.set_title(DISPLAY_NAMES.get(category, category.replace("_", " ").title()))
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("RMSE")
        ax.set_xticklabels([DISPLAY_NAMES.get(opt, opt) for opt in subset_sorted])
        ax.legend().remove()
        for i, opt in enumerate(subset_sorted):
            median_val = subset_medians.get(opt, np.nan)
            if not np.isnan(median_val):
                ax.text(
                    i + 0.15,
                    median_val,
                    f"{median_val:.3f}",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/rmse_syn.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_box_swarm()
    plot_best_fits()
    plot_parameter_recovery()
    plot_parameter_recovery_excluding_own()
    plot_synthetic_swarm()
