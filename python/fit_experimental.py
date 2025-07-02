#!/usr/bin/env python
"""Fit the AAV experimental data using the CMA-ES optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from objective_function import ObjectiveTracker
from optimizers import run_cma_es


_LOWER_BOUNDS = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.00005, 0.0001, 0.0001, 0.00005, 100, 0.00000001, 100, 100, 100, 0.00000001, 0.00000001, 0.5,0.0001,100];
_UPPER_BOUNDS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1000000, 0.00001, 1000000, 1000000, 1000000, 0.00001, 0.00001, 5,0.01,1000000];

def _load_aav_trace(mat_path: str) -> np.ndarray:
    """Load the AAV trace from a ``.mat`` file."""
    data = scipy.io.loadmat(mat_path)
    if "AAV" not in data:
        raise KeyError("AAV key not found in MATLAB file")
    return data["AAV"].flatten()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit experimental AAV data")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", default="results/aav_cmaes_history.csv", help="CSV output file"
    )
    parser.add_argument(
        "--data", default="data/experimental_data.mat", help="Path to the .mat file"
    )
    return parser.parse_args()


def main() -> None:
    """Run CMA-ES optimization on the experimental AAV data."""
    args = parse_args()

    trace = _load_aav_trace(args.data)
    base_params = pd.read_csv("data/ground_truth_params.csv")["parameter_value"].values

    opt_param_indices = list(range(len(base_params)))
    bounds = list(zip(_LOWER_BOUNDS, _UPPER_BOUNDS))

    objective = ObjectiveTracker(
        dataset_path=None,
        base_params=base_params,
        opt_param_indices=opt_param_indices,
        mean_trace=trace,
    )

    best_solution, best_sse = run_cma_es(
        objective_tracker=objective, bounds=bounds, random_seed=args.seed
    )

    Path("results").mkdir(exist_ok=True)
    history_df = pd.DataFrame(objective.history)
    history_df.to_csv(args.output, index=False, float_format="%.8f")

    print("\n--- Optimization Complete ---")
    print(f"Best SSE found: {best_sse:.4f}")
    print(f"History saved to {args.output}")


if __name__ == "__main__":
    main()
