"""Fit parameters to an experimental trace using various optimizers."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.io

from objective_function import ObjectiveTracker
from optimizers import (
    run_basin_hopping,
    run_cma_es,
    run_dual_annealing,
    run_lshade,
    run_pso,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed optimizer choice, seed, and variant.
    """
    parser = argparse.ArgumentParser(
        description="Optimize model parameters against experimental data"
    )
    parser.add_argument(
        "--optimizer",
        choices=[
            "cmaes",
            "lshade",
            "pso",
            "dual_annealing",
            "basin_hopping",
        ],
        default="cmaes",
        help=(
            "Which optimizer to use (cmaes, lshade, pso, "
            "dual_annealing, or basin_hopping)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--variant",
        choices=["AAV", "ASV", "LVA"],
        default="AAV",
        help="Experimental trace variant to fit",
    )
    return parser.parse_args()


def _load_trace(mat_path: str, variant: str) -> np.ndarray:
    """Load the experimental trace for the requested variant.

    Args:
        mat_path: Path to the ``.mat`` file containing traces.
        variant: Variant name (``AAV``, ``ASV``, or ``LVA``).

    Returns:
        One-dimensional numpy array of GFP measurements.
    """
    mat = scipy.io.loadmat(mat_path)
    trace = mat[variant].squeeze()
    return trace.astype(float)


def main() -> None:
    """Entry point for experimental data fitting."""
    args = parse_args()

    start_time = time.time()

    lower_bounds = [
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
    upper_bounds = [
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

    bounds = list(zip(lower_bounds, upper_bounds))

    trace = _load_trace("data/experimental_data.mat", args.variant)
    output_file = (
        f"results/{args.optimizer}_history_variant_"
        f"{args.variant}_run_{args.seed}.csv"
    )

    objective = ObjectiveTracker(
        dataset_path=None,
        mean_trace=trace,
    )

    if args.optimizer == "cmaes":
        best_solution, best_sse = run_cma_es(
            objective_tracker=objective,
            bounds=bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "lshade":
        best_solution, best_sse = run_lshade(
            objective_tracker=objective,
            bounds=bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "pso":
        best_solution, best_sse = run_pso(
            objective_tracker=objective,
            bounds=bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "dual_annealing":
        best_solution, best_sse = run_dual_annealing(
            objective_tracker=objective,
            bounds=bounds,
            random_seed=args.seed,
        )
    else:
        best_solution, best_sse = run_basin_hopping(
            objective_tracker=objective,
            bounds=bounds,
            random_seed=args.seed,
        )

    if not os.path.exists("results"):
        os.makedirs("results")

    history_df = pd.DataFrame(objective.history)
    history_df.to_csv(output_file, index=False, float_format="%.8f")

    duration_s = time.time() - start_time
    print("\n--- Optimization Complete ---")
    print(f"Total optimization time: {duration_s:.2f} seconds")
    print(f"Saved history to {output_file}")
    print(f"Best SSE found: {best_sse:.4f}")


if __name__ == "__main__":
    main()