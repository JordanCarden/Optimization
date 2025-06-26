"""Optimization utilities for running optimization algorithms.

This module provides wrappers for several optimizers configured for the
21-dimensional parameter space used throughout the repository.
"""

from __future__ import annotations

import cma
import numpy as np
from scipy.stats import qmc
from skopt import gp_minimize

from objective_function import ObjectiveTracker


def _lhs_samples(
    bounds: list[tuple[float, float]], n_samples: int, seed: int
) -> np.ndarray:
    """Return Latin Hypercube samples within the given bounds."""
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    sample = sampler.random(n=n_samples)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return qmc.scale(sample, lower, upper)


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
    x0 = x0 = _lhs_samples(bounds, 1, random_seed)[0]
    sigma0 = 0.25 * np.mean([ub - lb for lb, ub in bounds])
    popsize = 4 + int(3 * np.log(len(bounds)))

    options = {
        "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
        "maxfevals": 1000,
        "popsize": popsize,
        "seed": random_seed,
        "tolfun": 0,
        "tolx": 0
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
        n_calls=100,
        n_initial_points=30,
        initial_point_generator="lhs",
        acq_func="gp_hedge",
        random_state=random_seed,
    )

    return np.array(result.x), float(result.fun)


def run_lshade(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run the L-SHADE algorithm with a 5k evaluation budget."""
    from mealpy import FloatVar
    from mealpy.evolutionary_based.SHADE import L_SHADE

    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]

    var = FloatVar(lb=tuple(lower), ub=tuple(upper), name="param")

    budget = 1000
    calls = 0

    def wrapped(x: np.ndarray) -> float:
        nonlocal calls
        if calls >= budget:
            raise StopIteration("Evaluation budget reached")
        calls += 1
        return objective_tracker.evaluate(x)

    problem = {
        "obj_func": wrapped,
        "bounds": var,
        "minmax": "min",
        "log_to": None,
    }

    np.random.seed(random_seed)
    init_pop = _lhs_samples(bounds, 100, random_seed)
    model = L_SHADE(epoch=1000, pop_size=100, verbose=False)

    try:
        model.solve(problem, starting_solutions=init_pop)
    except StopIteration:
        pass

    return model.g_best.solution, float(model.g_best.target.fitness)


def run_pso(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run Particle Swarm Optimization with inertia weight scheduling."""
    import pyswarms as ps

    np.random.seed(random_seed)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    init_pos = _lhs_samples(bounds, 50, random_seed)

    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=50,
        dimensions=len(bounds),
        options=options,
        bounds=(lower, upper),
        init_pos=init_pos,
    )

    def _swarm_objective(x: np.ndarray) -> np.ndarray:
        """Evaluate a batch of particles using the single-vector objective."""
        costs = np.zeros(x.shape[0])
        for idx, particle in enumerate(x):
            costs[idx] = objective_tracker.evaluate(particle)
        return costs

    best_cost = np.inf
    best_pos = None
    for i in range(20):
        optimizer.options["w"] = 0.9 - 0.5 * (i / 99)
        cost, pos = optimizer.optimize(
            _swarm_objective,
            iters=1,
            verbose=False,
        )
        if cost < best_cost:
            best_cost, best_pos = cost, pos

    return np.array(best_pos), float(best_cost)


def run_direct(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run the DIRECT global optimization algorithm."""
    from scipy.optimize import direct

    result = direct(
        func=objective_tracker.evaluate,
        bounds=bounds,
        maxfun=1000,
        locally_biased=False,
    )

    return np.array(result.x), float(result.fun)
