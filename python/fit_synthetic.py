"""Entry point for running model parameter optimization."""

import os
import time
import argparse
import pandas as pd

from objective_function import ObjectiveTracker
from optimizers import (
    run_cma_es,
    run_lshade,
    run_pso,
    run_dual_annealing,
    run_basin_hopping,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed optimizer choice, seed, and dataset name.
    """
    parser = argparse.ArgumentParser(description="Run optimization routine")
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
            "Which optimizer to use (cmaes, lshade, pso, dual_annealing, or"
            " basin_hopping)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cmaes_1",
        help="Name of the dataset file to optimize on (e.g., 'cmaes_1')",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected optimization routine and save the evaluation history."""
    args = parse_args()

    total_run_start_time = time.time()

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

    dataset_path = f"data/{args.dataset}.csv"
    output_file = (
        f"results/{args.optimizer}_history_"
        f"{args.dataset}_run_{args.seed}.csv"
    )

    objective = ObjectiveTracker(
        dataset_path=dataset_path,
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

    total_run_end_time = time.time()
    total_duration_s = total_run_end_time - total_run_start_time

    print("\n--- Optimization Complete ---")
    print(f"Total optimization time: {total_duration_s:.2f} seconds")
    print(f"Saved history to {output_file}")
    print(f"Best SSE found: {best_sse:.4f}")


if __name__ == "__main__":
    main()