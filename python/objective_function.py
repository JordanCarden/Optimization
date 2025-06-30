"""Objective function wrapper for optimization."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from generate_data import MODEL_PARAMS
from simulate import simulate_variant_response


def unscale_parameters(normalized: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Convert normalized parameters back to their physical units.

    Args:
        normalized: Array of normalized parameters in ``[0, 1]``.
        bounds: Array with shape ``(n_params, 2)`` containing ``(lower, upper)``
            bounds for each parameter.

    Returns:
        Unscaled parameters corresponding to ``normalized``.
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + normalized * (upper - lower)


class ObjectiveTracker:
    """Evaluate SSE objective and track evaluation history.

    Evaluations compute the sum of squared errors (SSE) between observed data
    and model simulations. Each call is timed and stored in ``history``.
    """

    def __init__(
        self,
        dataset_path: str,
        base_params: np.ndarray,
        opt_param_indices: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> None:
        """Initialize the tracker and load the dataset.

        Args:
            dataset_path: Path to the CSV dataset file.
            base_params: Base parameter values.
            opt_param_indices: Indices of parameters to optimize.
        """
        dataset = pd.read_csv(dataset_path)
        traces = dataset[["replicate_1", "replicate_2", "replicate_3"]]
        self.mean_trace = traces.mean(axis=1).values

        self.base_params = base_params
        self.opt_param_indices = opt_param_indices
        self.bounds = np.array(bounds, dtype=float)

        self.call_count = 0
        self.history = []
        self.start_time = time.time()

    def evaluate(self, x: np.ndarray) -> float:
        """Compute SSE for a normalized parameter vector.

        Args:
            x: Normalized parameter values in ``[0, 1]``.

        Returns:
            float: Sum of squared errors between observed data and the
            simulated trace.
        """
        if self.call_count >= 110000:
            raise RuntimeError("Evaluation budget exceeded")

        self.call_count += 1

        params_to_sim = self.base_params.copy()
        unscaled = unscale_parameters(x, self.bounds)
        params_to_sim[self.opt_param_indices] = unscaled

        simulated = simulate_variant_response(
            params=params_to_sim,
            model_params=MODEL_PARAMS,
            variant="AAV",
        )

        if np.isnan(simulated).any():
            sse = 1e12
        else:
            sse = np.sum((self.mean_trace - simulated) ** 2)

        elapsed_seconds = time.time() - self.start_time

        self.history.append(
            {
                "params": [float(p) for p in unscaled],
                "sse": sse,
                "elapsed_time_s": elapsed_seconds,
            }
        )

        print(
            f"Call: {self.call_count}, SSE: {sse:.4f}, " f"Time: {elapsed_seconds:.2f}s"
        )
        return sse
