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
        0.065,      # params[0]: STAR/THS Txn. Rate (s⁻¹)
        0.001,      # params[1]: TetR Translation Rate (s⁻¹)
        0.065,      # params[2]: Y mRNA Txn. Rate (s⁻¹)
        0.065,      # params[3]: Z (GFP) mRNA Txn. Rate (s⁻¹)
        0.001,      # params[4]: GFP Translation Rate (s⁻¹)
        0.0008,     # params[5]: STAR Degradation Rate (s⁻¹)
        0.0008,     # params[6]: THS Degradation Rate (s⁻¹)
        0.0001,     # params[7]: TetR Degradation Rate (s⁻¹)
        0.0008,     # params[8]: Y mRNA Degradation Rate (s⁻¹)
        0.0008,     # params[9]: Z (GFP) mRNA Degradation Rate (s⁻¹)
        0.0001,     # params[10]: GFP "Degradation" Rate (s⁻¹)
        1.0e1,      # params[11]: STAR-Promoter Binding ((nM·s)⁻¹)
        1.0e-3,     # params[12]: STAR-Promoter Unbinding (s⁻¹)
        1.0e1,      # params[13]: THS-Y mRNA Binding ((nM·s)⁻¹)
        1.0e2,      # params[14]: TetR-DNA Binding ((nM·s)⁻¹)
        1.0e2,      # params[15]: TetR-aTc Binding ((nM·s)⁻¹)
        1.0e-4,     # params[16]: TetR-DNA Unbinding (s⁻¹)
        1.0e-2,     # params[17]: TetR-aTc Unbinding (s⁻¹)
        0.1,        # params[18]: GFP Scaling Factor (AU/nM) - Placeholder
        0.0008,     # params[19]: Y_active Degradation (s⁻¹)
        0.0         # params[20]: TetR-Pz_active Binding ((nM·s)⁻¹)
    ]

    upper_bounds = [
        0.5,        # params[0]: STAR/THS Txn. Rate (s⁻¹)
        0.5,        # params[1]: TetR Translation Rate (s⁻¹)
        0.5,        # params[2]: Y mRNA Txn. Rate (s⁻¹)
        0.5,        # params[3]: Z (GFP) mRNA Txn. Rate (s⁻¹)
        0.5,        # params[4]: GFP Translation Rate (s⁻¹)
        0.004,      # params[5]: STAR Degradation Rate (s⁻¹)
        0.004,      # params[6]: THS Degradation Rate (s⁻¹)
        0.0004,     # params[7]: TetR Degradation Rate (s⁻¹)
        0.004,      # params[8]: Y mRNA Degradation Rate (s⁻¹)
        0.004,      # params[9]: Z (GFP) mRNA Degradation Rate (s⁻¹)
        0.0004,     # params[10]: GFP "Degradation" Rate (s⁻¹)
        1.0e6,      # params[11]: STAR-Promoter Binding ((nM·s)⁻¹)
        1.0,        # params[12]: STAR-Promoter Unbinding (s⁻¹)
        1.0e6,      # params[13]: THS-Y mRNA Binding ((nM·s)⁻¹)
        1.0e4,      # params[14]: TetR-DNA Binding ((nM·s)⁻¹)
        1.0e5,      # params[15]: TetR-aTc Binding ((nM·s)⁻¹)
        0.1,        # params[16]: TetR-DNA Unbinding (s⁻¹)
        1.0,        # params[17]: TetR-aTc Unbinding (s⁻¹)
        10.0,       # params[18]: GFP Scaling Factor (AU/nM) - Placeholder
        0.004,      # params[19]: Y_active Degradation (s⁻¹)
        1.0e4       # params[20]: TetR-Pz_active Binding ((nM·s)⁻¹)
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
