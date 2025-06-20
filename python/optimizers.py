import math
import numpy as np
import cma
from scipy.optimize import direct
from skopt import gp_minimize
from mealpy.evolutionary_based.SHADE import L_SHADE
from pyswarms.single.global_best import GlobalBestPSO

from simulate import simulate_variant_response


def objective_function(candidate_params, base_params,
                       model_params, experimental_data,
                       variant, opt_param_indices):
    """Compute sum of squared errors between simulation and experiment.

    Args:
        candidate_params (np.ndarray): Parameters to optimize.
        base_params (np.ndarray): Full parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        opt_param_indices (Sequence[int]): Indices of params to replace.

    Returns:
        float: SSE or infinity if output sizes mismatch.
    """
    trial = base_params.copy()
    trial[opt_param_indices] = candidate_params
    sim = simulate_variant_response(trial, model_params, variant)
    exp = experimental_data[variant]
    if sim.size != exp.size:
        return float(np.inf)
    return float(np.sum((sim - exp) ** 2))


def run_bayesian_optimization(base_params, model_params,
                              experimental_data, variant, bounds,
                              opt_param_indices, random_seed):
    """Optimize using Bayesian Optimization with a 5 000-call budget.

    Args:
        base_params (np.ndarray): Initial parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        bounds (Sequence[tuple]): Parameter bounds.
        opt_param_indices (Sequence[int]): Indices of params to update.
        random_seed (int): Random seed.

    Returns:
        np.ndarray: Full parameter vector with optimized values.
    """
    result = gp_minimize(
        func=lambda x: objective_function(
            x, base_params, model_params, experimental_data,
            variant, opt_param_indices
        ),
        dimensions=bounds,
        acq_func='gp_hedge',
        n_calls=5000,
        n_initial_points=30,
        initial_point_generator='lhs',
        random_state=random_seed,
    )
    optimized = base_params.copy()
    optimized[opt_param_indices] = result.x
    return optimized


def run_cma_es(base_params, model_params, experimental_data,
               variant, bounds, opt_param_indices, random_seed):
    """Optimize using CMA-ES with a 5 000-call budget.

    Args:
        base_params (np.ndarray): Initial parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        bounds (Sequence[tuple]): Parameter bounds.
        opt_param_indices (Sequence[int]): Indices of params to update.
        random_seed (int): Random seed.

    Returns:
        np.ndarray: Full parameter vector with optimized values.
    """
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])
    sigma0 = 0.25 * np.mean([ub - lb for lb, ub in bounds])
    popsize = 4 + int(3 * np.log(len(bounds)))
    counter = {'count': 0}

    def wrapped_obj(x):
        counter['count'] += 1
        return objective_function(
            x, base_params, model_params, experimental_data,
            variant, opt_param_indices
        )

    options = {
        'bounds': [lower_bounds, upper_bounds],
        'maxfevals': 5000,
        'popsize': popsize,
        'seed': random_seed,
    }
    solution, _ = cma.fmin2(wrapped_obj, x0, sigma0, options=options)
    optimized = base_params.copy()
    optimized[opt_param_indices] = solution
    return optimized


def run_lshade(base_params, model_params, experimental_data,
               variant, bounds, opt_param_indices):
    """Optimize using L-SHADE with a 5 000-call budget.

    Args:
        base_params (np.ndarray): Initial parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        bounds (Sequence[tuple]): Parameter bounds.
        opt_param_indices (Sequence[int]): Indices of params to update.

    Returns:
        np.ndarray: Full parameter vector with optimized values.
    """
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    pop_size = 100
    epoch = math.ceil(5000 / pop_size)

    def wrapped_obj(sol):
        return objective_function(
            sol, base_params, model_params, experimental_data,
            variant, opt_param_indices
        )

    problem = {
        'fit_func': wrapped_obj,
        'lb': lb,
        'ub': ub,
        'minmax': 'min',
    }
    model = L_SHADE(epoch=epoch, pop_size=pop_size)
    model.solve(problem)
    optimized = base_params.copy()
    optimized[opt_param_indices] = model.g_best.solution
    return optimized


def run_pso(base_params, model_params, experimental_data, variant,
            bounds, opt_param_indices):
    """Optimize using Particle Swarm Optimization (5 000-call budget).

    Args:
        base_params (np.ndarray): Initial parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        bounds (Sequence[tuple]): Parameter bounds.
        opt_param_indices (Sequence[int]): Indices of params to update.

    Returns:
        np.ndarray: Full parameter vector with optimized values.
    """
    n_particles = 50
    n_iterations = 100
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds),
        options=options,
        bounds=(lb, ub),
    )

    def pso_obj(x):
        return np.array([
            objective_function(
                x[i], base_params, model_params, experimental_data,
                variant, opt_param_indices
            ) for i in range(x.shape[0])
        ])

    for i in range(n_iterations):
        optimizer.options['w'] = 0.9 - 0.5 * i / (n_iterations - 1)
        optimizer.optimize(pso_obj, iters=1, n_processes=None)

    best_pos = optimizer.best_pos
    optimized = base_params.copy()
    optimized[opt_param_indices] = best_pos
    return optimized


def run_direct(base_params, model_params, experimental_data,
               variant, bounds, opt_param_indices):
    """Optimize using the DIRECT algorithm with a 5 000-call budget.

    Args:
        base_params (np.ndarray): Initial parameter vector.
        model_params (dict): Simulation settings.
        experimental_data (dict): Observed data per variant.
        variant (str): Variant identifier.
        bounds (Sequence[tuple]): Parameter bounds.
        opt_param_indices (Sequence[int]): Indices of params to update.

    Returns:
        np.ndarray: Full parameter vector with optimized values.
    """
    counter = {'count': 0}

    def wrapped_obj(x):
        counter['count'] += 1
        return objective_function(
            x, base_params, model_params, experimental_data,
            variant, opt_param_indices
        )

    result = direct(
        wrapped_obj, bounds=bounds, maxfun=5000, locally_biased=False
    )
    optimized = base_params.copy()
    optimized[opt_param_indices] = result.x
    return optimized
