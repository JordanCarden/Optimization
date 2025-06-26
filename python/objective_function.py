"""Objective function wrapper for optimization."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from generate_data import MODEL_PARAMS
from simulate import simulate_variant_response


class ObjectiveTracker:
    """Evaluate SSE objective and track evaluation history, including timing.

    Evaluations compute the sum of squared errors (SSE) between observed
    data and model simulations. Each evaluation is logged with elapsed time.
    """

    def __init__(
        self,
        dataset_path: str,
        base_params: np.ndarray,
        opt_param_indices: np.ndarray,
    ):
        """Initializes the tracker and records the start time.

        Args:
            dataset_path (str): Path to the CSV dataset file.
            base_params (sequence of float): Base parameter values.
            opt_param_indices (sequence of int): Indices of parameters to
                optimize.
        """
        dataset = pd.read_csv(dataset_path)
        traces = dataset[['replicate_1', 'replicate_2', 'replicate_3']]
        self.mean_trace = traces.mean(axis=1).values

        self.base_params = base_params
        self.opt_param_indices = opt_param_indices

        self.call_count = 0
        self.history = []
        self.start_time = time.time()

    def evaluate(self, x: np.ndarray) -> float:
        """Compute SSE and log the result along with elapsed time.

        Args:
            x (sequence of float): Parameter values to substitute into the
                base parameters.

        Returns:
            float: Sum of squared errors (SSE) between observed data and
                simulated trace.
        """
        if self.call_count >= 100000:
            raise RuntimeError('Evaluation budget exceeded')

        self.call_count += 1

        params_to_sim = self.base_params.copy()
        params_to_sim[self.opt_param_indices] = x

        simulated = simulate_variant_response(
            params=params_to_sim,
            model_params=MODEL_PARAMS,
            variant='AAV',
        )

        if np.isnan(simulated).any():
            sse = 1e12
        else:
            sse = np.sum((self.mean_trace - simulated) ** 2)

        elapsed_seconds = time.time() - self.start_time

        self.history.append({
            'params': [float(p) for p in x],
            'sse': sse,
            'elapsed_time_s': elapsed_seconds,
        })

        print(
            f'Call: {self.call_count}, SSE: {sse:.4f}, '
            f'Time: {elapsed_seconds:.2f}s'
        )
        return sse
