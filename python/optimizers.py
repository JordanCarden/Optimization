"""Wrappers for the optimization algorithms used throughout the repository."""

from __future__ import annotations

import cma
import numpy as np
from scipy.stats import qmc
from skopt import gp_minimize

from objective_function import ObjectiveTracker


def _lhs_samples(
    bounds: list[tuple[float, float]], n_samples: int, seed: int
) -> np.ndarray:
    """Generate Latin Hypercube samples.

    Args:
        bounds: List of ``(lower, upper)`` tuples for each dimension.
        n_samples: Number of points to sample.
        seed: Random seed for the sampler.

    Returns:
        Array of shape ``(n_samples, len(bounds))`` with sampled points.
    """
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
        bounds: Sequence of ``(lower, upper)`` bounds for each parameter.
        random_seed: Seed for the CMA-ES optimizer.

    Returns:
        Tuple ``(best_params, best_sse)`` with the optimal parameters and SSE.
    """
    x0 = _lhs_samples(bounds, 1, random_seed)[0]
    sigma0 = 0.25 * np.mean([ub - lb for lb, ub in bounds])
    popsize = 4 + int(3 * np.log(len(bounds)))

    options = {
        "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
        "maxfevals": 21000,
        "popsize": popsize,
        "seed": random_seed,
        "tolfun": 0,
        "tolx": 0,
        "tolstagnation": 0,
        "verb_log": 0,
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
        Tuple ``(best_params, best_sse)`` with the optimal parameters and SSE.
    """
    result = gp_minimize(
        func=objective_tracker.evaluate,
        dimensions=bounds,
        n_calls=21000,
        n_initial_points=210,
        initial_point_generator="lhs",
        acq_func="EI",
        base_estimator="ET",
        random_state=random_seed,
    )

    return np.array(result.x), float(result.fun)


def run_lshade(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run the L-SHADE algorithm with a 5k evaluation budget.

    Args:
        objective_tracker: Tracker providing the evaluate method.
        bounds: Sequence of ``(lower, upper)`` bounds for each parameter.
        random_seed: Seed for the optimizer.

    Returns:
        Tuple ``(best_params, best_sse)`` with the optimal parameters and SSE.
    """
    from mealpy import FloatVar
    from mealpy.evolutionary_based.SHADE import L_SHADE

    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]

    var = FloatVar(lb=tuple(lower), ub=tuple(upper), name="param")

    budget = 21000
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
    model = L_SHADE(epoch=10000, pop_size=100, verbose=False)

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
    """Run Particle Swarm Optimization with Clerc constriction parameters.

    Args:
        objective_tracker: Tracker providing the evaluate method.
        bounds: Sequence of ``(lower, upper)`` bounds for each parameter.
        random_seed: Seed for the optimizer.

    Returns:
        Tuple ``(best_params, best_sse)`` with the optimal parameters and SSE.
    """
    import pyswarms as ps

    np.random.seed(random_seed)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    init_pos = _lhs_samples(bounds, 50, random_seed)

    options = {"c1": 1.49445, "c2": 1.49445, "w": 0.729}
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

    cost, pos = optimizer.optimize(
        _swarm_objective,
        iters=420,
        verbose=False,
    )

    return np.array(pos), float(cost)


def run_direct(
    objective_tracker: ObjectiveTracker,
    bounds: list[tuple[float, float]],
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """Run the DIRECT global optimization algorithm.

    Args:
        objective_tracker: Tracker providing the evaluate method.
        bounds: Sequence of ``(lower, upper)`` bounds for each parameter.
        random_seed: Seed for the optimizer.

    Returns:
        Tuple ``(best_params, best_sse)`` with the optimal parameters and SSE.
    """
    from scipy.optimize import direct

    result = direct(
        func=objective_tracker.evaluate,
        bounds=bounds,
        maxfun=21000,
        locally_biased=False,
    )

    return np.array(result.x), float(result.fun)
