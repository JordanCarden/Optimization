"""Generate plots comparing optimizer performance on experimental data."""

from __future__ import annotations

import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simulate_model import simulate_model  # type: ignore
from data_utils import get_experimental_data  # type: ignore


Algorithm = str


BEST_PARAM_FILES: dict[Algorithm, str] = {
    "cmaes": "data/cmaes_best_params.csv",
    "lshade": "data/lshade_best_params.csv",
    "pso": "data/pso_best_params.csv",
    "dual_annealing": "data/dual_annealing_best_params.csv",
    "basin_hopping": "data/basin_hopping_best_params.csv",
}


def _load_best_params(file_path: str) -> pd.Series:
    """Load a single row of best-fit parameters from ``file_path``."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"No parameters found in {file_path}")
    return df.iloc[0]


def _rmse(sse: float, n: int) -> float:
    """Return RMSE given SSE and sample size."""
    return math.sqrt(sse / n)


def _best_fit_plot() -> None:
    """Generate best-fit curves overlaid on experimental data."""
    exp_df = get_experimental_data()
    concentrations = exp_df["concentration"].to_numpy()
    n_points = len(exp_df)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="concentration",
        y="signal",
        data=exp_df,
        color="black",
        label="Experimental",
    )

    for algo, path in BEST_PARAM_FILES.items():
        params_df = pd.read_csv(path)
        params = params_df.iloc[0].to_numpy(dtype=float)
        sse = params_df.get("sse", pd.Series([float("nan")])).iloc[0]
        simulated = simulate_model(params, concentrations)
        if math.isnan(sse):
            sse = float(((exp_df["signal"] - simulated) ** 2).sum())
        rmse = _rmse(sse, n_points)
        label = f"{algo.replace('_', ' ').title()} (RMSE: {rmse:.2f})"
        sns.lineplot(x=concentrations, y=simulated, label=label)

    plt.xscale("log")
    plt.xlabel("Concentration")
    plt.ylabel("Signal")
    plt.title("Best Model Fit vs. Experimental Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "best_fit_lines.png"), dpi=300)
    plt.close()


def _collect_best_rmse(results_dir: str, n_points: int) -> pd.DataFrame:
    """Return a DataFrame of best RMSE from optimizer result files."""
    records: list[dict[str, float | str]] = []
    for csv_path in glob.glob(os.path.join(results_dir, "*.csv")):
        df = pd.read_csv(csv_path)
        min_sse = df["sse"].min()
        algo = os.path.basename(csv_path).split("_")[0]
        rmse = _rmse(min_sse, n_points)
        records.append({"Algorithm": algo, "Best RMSE": rmse})
    return pd.DataFrame.from_records(records)


def _rmse_distribution_plot(results_dir: str, n_points: int) -> None:
    """Plot RMSE distribution across multiple runs."""
    df = _collect_best_rmse(results_dir, n_points)
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x="Algorithm", y="Best RMSE", palette="colorblind")
    plt.title("Distribution of Best RMSE Across 13 Independent Runs")
    plt.xlabel("Algorithm")
    plt.ylabel("Best RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "rmse_distribution_swarm.png"), dpi=300)
    plt.close()


def main() -> None:
    """Entry point for plot generation."""
    os.makedirs("plots", exist_ok=True)
    exp_df = get_experimental_data()
    n_points = len(exp_df)
    _best_fit_plot()
    _rmse_distribution_plot("experimental_results", n_points)


if __name__ == "__main__":
    main()
