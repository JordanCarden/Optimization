"""Entry point for running model parameter optimization."""

import argparse
import os
import time
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
    """Parse command-line arguments for optimizer selection.

    Returns:
        argparse.Namespace: Parsed arguments containing only the optimizer
        choice.
    """
    parser = argparse.ArgumentParser(description="Run optimization routine")
    parser.add_argument(
        "--optimizer",
        choices=["cmaes", "bo", "lshade", "pso", "direct"],
        default="cmaes",
        help="Which optimizer to use (cmaes, bo, lshade, pso, or direct)",
    )
    return parser.parse_args()


def main() -> None:
    """Run optimizations for all datasets and seeds."""
    args = parse_args()

    total_run_start_time = time.time()

    base_params = pd.read_csv(
        "data/ground_truth_params.csv"
    )["parameter_value"].values

    lower_bounds = [
        0.1, 0.0005, 1.0, 0.5, 1.0,
        0.01, 0.01, 0.001, 0.01, 0.01,
        0.01, 100, 1e-5, 100, 100, 100,
        1e-8, 1e-8, 0.5, 0.0001, 100,
    ]
    upper_bounds = [
        50.0, 0.5, 100.0, 50.0, 100.0,
        1.0, 1.0, 0.1, 1.0, 1.0,
        1.0, 1e6, 1e-3, 1e6, 1e6, 1e6,
        1e-5, 1e-5, 5, 0.01, 1e6,
    ]
    bounds = list(zip(lower_bounds, upper_bounds))
    opt_param_indices = list(range(len(base_params)))
    opt_bounds = [bounds[i] for i in opt_param_indices]

    if not os.path.exists("results"):
        os.makedirs("results")

    for dataset in [1, 2, 3]:
        for seed in [1, 2, 3]:
            dataset_path = f"data/dataset_{dataset}.csv"
            output_file = (
                f"results/{args.optimizer}_history_"
                f"dataset_{dataset}_run_{seed}.csv"
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
                    random_seed=seed,
                )
            elif args.optimizer == "cmaes":
                best_solution, best_sse = run_cma_es(
                    objective_tracker=objective,
                    bounds=opt_bounds,
                    random_seed=seed,
                )
            elif args.optimizer == "lshade":
                best_solution, best_sse = run_lshade(
                    objective_tracker=objective,
                    bounds=opt_bounds,
                    random_seed=seed,
                )
            elif args.optimizer == "pso":
                best_solution, best_sse = run_pso(
                    objective_tracker=objective,
                    bounds=opt_bounds,
                    random_seed=seed,
                )
            else:
                best_solution, best_sse = run_direct(
                    objective_tracker=objective,
                    bounds=opt_bounds,
                    random_seed=seed,
                )

            final_params = base_params.copy()
            final_params[opt_param_indices] = best_solution

            history_df = pd.DataFrame(objective.history)
            history_df.to_csv(output_file, index=False, float_format="%.8f")

            print("\n--- Optimization Complete ---")
            print(
                f"Dataset: {dataset}, Seed: {seed}, Best SSE: {best_sse:.4f}"
            )
            print(f"Saved history to {output_file}")

    total_run_end_time = time.time()
    total_duration_s = total_run_end_time - total_run_start_time

    print("\nAll optimizations finished")
    print(f"Total time for all runs: {total_duration_s:.2f} seconds")


if __name__ == "__main__":
    main()
