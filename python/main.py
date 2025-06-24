import os
import time
import numpy as np
import pandas as pd

from objective_function import ObjectiveTracker
from optimizers import run_cma_es


def main() -> None:
    """Run CMA-ES optimization and save evaluation history.

    Returns:
        None
    """
    total_run_start_time = time.time()

    base_params = pd.read_csv(
        'data/ground_truth_params.csv'
    )['parameter_value'].values

    lower_bounds = [
        0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
        0.0001, 0.0001, 0.00005, 0.0001, 0.0001,
        0.00005, 100, 1e-8, 100, 100, 100,
        1e-8, 1e-8, 0.5, 0.0001, 100,
    ]
    upper_bounds = [
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 1e6, 1e-5, 1e6, 1e6, 1e6,
        1e-5, 1e-5, 5, 0.01, 1e6,
    ]
    bounds = list(zip(lower_bounds, upper_bounds))
    opt_param_indices = list(range(len(base_params)))
    opt_bounds = [bounds[i] for i in opt_param_indices]

    dataset_path = 'data/dataset_1.csv'
    run_seed = 1
    output_file = (
        f'results/cma_es_history_dataset_1_run_{run_seed}.csv'
    )

    objective = ObjectiveTracker(
        dataset_path=dataset_path,
        base_params=base_params,
        opt_param_indices=opt_param_indices,
    )
    best_solution, best_sse = run_cma_es(
        objective_tracker=objective,
        bounds=opt_bounds,
        random_seed=run_seed,
    )

    final_params = base_params.copy()
    final_params[opt_param_indices] = best_solution

    if not os.path.exists('results'):
        os.makedirs('results')
    history_df = pd.DataFrame(objective.history)
    history_df.to_csv(output_file, index=False)

    total_run_end_time = time.time()
    total_duration_s = total_run_end_time - total_run_start_time

    print('\n--- Optimization Complete ---')
    print(f'Total optimization time: {total_duration_s:.2f} seconds')
    print(f'Saved history to {output_file}')
    print(f'Best SSE found: {best_sse:.4f}')
