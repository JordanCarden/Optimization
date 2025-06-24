"""Optimization utilities for running optimization algorithms.

This module currently provides wrappers for CMA-ES and Bayesian Optimization
configured for the 21-dimensional search space described in the technical
specification.
"""

from __future__ import annotations

import cma
import numpy as np
from skopt import gp_minimize

from objective_function import ObjectiveTracker


def run_cma_es(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run CMA-ES on the provided objective.

    Args:
        objective_tracker: Tracker providing the evaluate method.
        bounds: Sequence of (lower, upper) bounds for each parameter.
        random_seed: Seed for the CMA-ES optimizer.

    Returns:
        Tuple of best-found parameter vector and its SSE value.
    """
    x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])
    sigma0 = 0.25 * np.mean([ub - lb for lb, ub in bounds])
    popsize = 4 + int(3 * np.log(len(bounds)))

    options = {
        "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
        "maxfevals": 5000,
        "popsize": popsize,
        "seed": random_seed,
    }

    solution, es = cma.fmin2(
        objective_tracker.evaluate,
        x0,
        sigma0,
        options=options,
    )

    return solution, es.best.f


def run_bayesian_optimization(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run Bayesian Optimization using scikit-optimize.

    Args:
        objective_tracker: Tracker providing the evaluate method.
        bounds: Sequence of ``(lower, upper)`` bounds for each parameter.
        random_seed: Seed for the optimizer.

    Returns:
        Tuple of best-found parameter vector and its SSE value.
    """
    result = gp_minimize(
        func=objective_tracker.evaluate,
        dimensions=bounds,
        n_calls=5000,
        n_initial_points=30,
        initial_point_generator="lhs",
        acq_func="gp_hedge",
        random_state=random_seed,
    )

    return np.array(result.x), float(result.fun)
