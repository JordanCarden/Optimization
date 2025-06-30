"""Entry point for running model parameter optimization."""

import os
import time
import argparse
import pandas as pd

from objective_function import ObjectiveTracker
from optimizers import (
    run_bayesian_optimization,
    run_cma_es,
    run_lshade,
    run_pso,
    run_direct,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed optimizer choice, seed, and dataset number.
    """
    parser = argparse.ArgumentParser(description="Run optimization routine")
    parser.add_argument(
        "--optimizer",
        choices=["cmaes", "bo", "lshade", "pso", "direct"],
        default="cmaes",
        help=("Which optimizer to use (cmaes, bo, lshade, pso, or direct)"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=1,
        help="Dataset number to optimize on",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected optimization routine and save the evaluation history."""
    args = parse_args()

    total_run_start_time = time.time()

    base_params = pd.read_csv("data/ground_truth_params.csv")["parameter_value"].values

    lower_bounds = [

    ]
    upper_bounds = [

    ]


    bounds = list(zip(lower_bounds, upper_bounds))
    opt_param_indices = list(range(len(base_params)))
    opt_bounds = [bounds[i] for i in opt_param_indices]

    dataset_path = f"data/dataset_{args.dataset}.csv"
    output_file = (
        f"results/{args.optimizer}_history_"
        f"dataset_{args.dataset}_run_{args.seed}.csv"
    )

    objective = ObjectiveTracker(
        dataset_path=dataset_path,
        base_params=base_params,
        opt_param_indices=opt_param_indices,
    )
    if args.optimizer == "bo":
        best_solution, best_sse = run_bayesian_optimization(
            objective_tracker=objective,
            bounds=opt_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "cmaes":
        best_solution, best_sse = run_cma_es(
            objective_tracker=objective,
            bounds=opt_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "lshade":
        best_solution, best_sse = run_lshade(
            objective_tracker=objective,
            bounds=opt_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "pso":
        best_solution, best_sse = run_pso(
            objective_tracker=objective,
            bounds=opt_bounds,
            random_seed=args.seed,
        )
    else:
        best_solution, best_sse = run_direct(
            objective_tracker=objective,
            bounds=opt_bounds,
            random_seed=args.seed,
        )

    final_params = base_params.copy()
    final_params[opt_param_indices] = best_solution

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
