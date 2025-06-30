"""Entry point for running model parameter optimization."""

import os
import time
import argparse
import numpy as np
import pandas as pd

from objective_function import ObjectiveTracker, unscale_parameters
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
        0.0470940318,
        0.00482725805,
        0.101149449,
        0.0845102567,
        0.373637489,
        0.00216021069,
        0.00305100532,
        0.000376967367,
        0.00129482278,
        0.00427994896,
        0.00560767076,
        103.341863,
        0.00000263288502,
        57.5797013,
        83.592078,
        101.59061,
        0.000119814795,
        0.000078378013,
        0.00361074026,
        0.00000160724775,
        29.3045286,
    ]
    upper_bounds = [
        470.940318,
        48.2725805,
        1011.49449,
        845.102567,
        3736.37489,
        21.6021069,
        30.5100532,
        3.76967367,
        12.9482278,
        42.7994896,
        56.0767076,
        1033418.63,
        0.0263288502,
        575797.013,
        835920.78,
        1015906.1,
        1.19814795,
        0.78378013,
        36.1074026,
        0.0160724775,
        293045.286,
    ]


    bounds = list(zip(lower_bounds, upper_bounds))
    opt_param_indices = list(range(len(base_params)))
    opt_bounds = [bounds[i] for i in opt_param_indices]
    norm_bounds = [(0.0, 1.0)] * len(opt_bounds)

    dataset_path = f"data/dataset_{args.dataset}.csv"
    output_file = (
        f"results/{args.optimizer}_history_"
        f"dataset_{args.dataset}_run_{args.seed}.csv"
    )

    objective = ObjectiveTracker(
        dataset_path=dataset_path,
        base_params=base_params,
        opt_param_indices=opt_param_indices,
        bounds=opt_bounds,
    )
    if args.optimizer == "bo":
        best_solution, best_sse = run_bayesian_optimization(
            objective_tracker=objective,
            bounds=norm_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "cmaes":
        best_solution, best_sse = run_cma_es(
            objective_tracker=objective,
            bounds=norm_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "lshade":
        best_solution, best_sse = run_lshade(
            objective_tracker=objective,
            bounds=norm_bounds,
            random_seed=args.seed,
        )
    elif args.optimizer == "pso":
        best_solution, best_sse = run_pso(
            objective_tracker=objective,
            bounds=norm_bounds,
            random_seed=args.seed,
        )
    else:
        best_solution, best_sse = run_direct(
            objective_tracker=objective,
            bounds=norm_bounds,
            random_seed=args.seed,
        )

    final_params = base_params.copy()
    final_params[opt_param_indices] = unscale_parameters(best_solution, np.array(opt_bounds))

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
