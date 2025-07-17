"""Generate swarm plots for synthetic benchmark results."""

from __future__ import annotations

import ast
import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

# Constants
RESULTS_DIR = "synthetic_results"
AGGREGATE_FILE = "synthetic_results.csv"
DATA_DIR = "data"
ALGORITHMS = sorted([
    "basin_hopping",
    "cmaes",
    "dual_annealing",
    "lshade",
    "pso",
])
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}
CUSTOM_COLORS = ["#FFBE0B", "#FB5607", "#FF006E", "#8338EC", "#3A86FF"]
COLOR_MAP = dict(zip(ALGORITHMS, CUSTOM_COLORS))


def get_experimental_data() -> np.ndarray:
    """Load the experimental AAV trace from ``DATA_DIR``."""
    mat_path = os.path.join(DATA_DIR, "experimental_data.mat")
    return loadmat(mat_path)["AAV"].ravel().astype(float)


def _iter_result_files() -> Iterable[str]:
    """Yield result file paths from ``RESULTS_DIR`` or the aggregate file."""
    if os.path.isdir(RESULTS_DIR):
        for file_name in os.listdir(RESULTS_DIR):
            if file_name.endswith(".csv"):
                yield os.path.join(RESULTS_DIR, file_name)
    elif os.path.exists(AGGREGATE_FILE):
        yield AGGREGATE_FILE
    else:
        raise FileNotFoundError("Synthetic results not found")


def _load_results() -> pd.DataFrame:
    """Load all synthetic run results into a single ``DataFrame``."""
    dfs: List[pd.DataFrame] = []
    for path in _iter_result_files():
        if path == AGGREGATE_FILE:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path)
            run_algo = os.path.basename(path).split("_")[0]
            df.insert(0, "optimizer", run_algo)
            dataset = "_".join(os.path.basename(path).split("_")[2:4])
            df.insert(1, "dataset", dataset)
        dfs.append(df)
    if not dfs:
        raise ValueError("No result files loaded")
    return pd.concat(dfs, ignore_index=True)


def _prepare_metrics(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    """Compute RMSE and MASE metrics for each run."""
    rmse_list: List[float] = []
    mase_list: List[float] = []
    algorithms: List[str] = []
    for _, row in df.iterrows():
        algo = row["optimizer"]
        sse = float(row["min_sse"])
        rmse = np.sqrt(sse / n_points)
        rmse_list.append(rmse)

        recovered = np.array(ast.literal_eval(row["params"]), dtype=float)
        source = str(row["dataset"]).rsplit("_", 1)[0]
        true_path = os.path.join(DATA_DIR, f"{source}_best_params.csv")
        true_params = pd.read_csv(true_path).iloc[0].to_numpy(dtype=float)
        min_val = true_params.min()
        max_val = true_params.max()
        scaled_rec = (recovered - min_val) / (max_val - min_val)
        scaled_true = (true_params - min_val) / (max_val - min_val)
        mase = float(np.mean(np.abs(scaled_rec - scaled_true)))
        mase_list.append(mase)
        algorithms.append(DISPLAY_NAMES.get(algo, algo))

    return pd.DataFrame(
        {
            "Algorithm": algorithms,
            "Final RMSE": rmse_list,
            "Parameter Recovery MASE": mase_list,
        }
    )


def _save_swarm_plot(
    data: pd.DataFrame,
    value_col: str,
    y_label: str,
    file_name: str,
) -> None:
    """Save a swarm plot for the provided metric."""
    display_color_map = {DISPLAY_NAMES[a]: COLOR_MAP[a] for a in ALGORITHMS}
    display_order = [DISPLAY_NAMES[a] for a in ALGORITHMS]

    plt.figure(figsize=(10, 7))
    sns.swarmplot(
        data=data,
        x="Algorithm",
        y=value_col,
        palette=display_color_map,
        order=display_order,
        size=5,
    )
    plt.ylabel(y_label)
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()


def main() -> None:
    """Generate swarm plots for synthetic RMSE and MASE."""
    num_points = len(get_experimental_data())
    df = _load_results()
    metrics = _prepare_metrics(df, num_points)
    os.makedirs("plots", exist_ok=True)
    _save_swarm_plot(
        metrics,
        value_col="Final RMSE",
        y_label="Final RMSE",
        file_name=os.path.join("plots", "synthetic_rmse_swarm.png"),
    )
    _save_swarm_plot(
        metrics,
        value_col="Parameter Recovery MASE",
        y_label="Parameter Recovery MASE",
        file_name=os.path.join("plots", "parameter_recovery_mase_swarm.png"),
    )


if __name__ == "__main__":
    main()
