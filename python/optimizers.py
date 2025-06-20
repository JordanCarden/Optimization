# python/optimizers.py

import numpy as np
from scipy.optimize import differential_evolution
import cma
from skopt import gp_minimize
from pyswarms.single.global_best import GlobalBestPSO

from simulate import simulate_variant_response

OPT_PARAM_INDICES = []


def objective_function(candidate_params,
                       base_params,
                       model_params,
                       experimental_data,
                       variant):
    """Compute SSE between simulation and experiment.

    Args:
        candidate_params: np.ndarray, values for optimized indices.
        base_params: np.ndarray, original parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict[str, np.ndarray], maps variants to data.
        variant: str, one of simulate.VARIANTS.

    Returns:
        float, sum of squared errors. Returns np.inf on size mismatch.
    """
    trial_params = base_params.copy()
    trial_params[OPT_PARAM_INDICES] = candidate_params
    sim = simulate_variant_response(trial_params, model_params, variant)
    exp = experimental_data[variant]
    if sim.size != exp.size:
        return np.inf
    return float(np.sum((sim - exp) ** 2))


def run_cma_es(base_params, model_params, experimental_data, variant):
    """Optimize parameters using CMA-ES.

    CMA-ES is an evolutionary algorithm well-suited for difficult,
    non-convex, and rugged landscapes. It adapts the covariance matrix
    of a multivariate normal distribution to generate candidate solutions,
    effectively learning the shape of the search space.

    Args:
        base_params: np.ndarray, initial parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict, maps variants to data.
        variant: str, the variant to optimize for.

    Returns:
        np.ndarray, optimized parameter vector.
    """
    initial_guess = base_params[OPT_PARAM_INDICES]
    sigma0 = 0.25
    bounds = [
        [0 for _ in OPT_PARAM_INDICES],
        [max(p * 100, 1e-9) for p in initial_guess]
    ]

    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, {'bounds': bounds})
    

    obj_func = lambda x: objective_function(
        x, base_params, model_params, experimental_data, variant
    )

    es.optimize(obj_func, iterations=100)
    
    optimized = base_params.copy()
    optimized[OPT_PARAM_INDICES] = es.result.xbest
    return optimized


def run_lshade(base_params, model_params, experimental_data, variant):
    """Optimize using L-SHADE, an adaptive Differential Evolution.

    L-SHADE (Limited-memory, Success-History based Adaptive
    Differential Evolution) automatically adapts the mutation (F) and
    crossover (CR) parameters based on their historical success,
    which can improve performance on complex problems.

    Args:
        base_params: np.ndarray, initial parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict, maps variants to data.
        variant: str, the variant to optimize for.

    Returns:
        np.ndarray, optimized parameter vector.
    """
    bounds = [
        (0, max(base_params[i] * 100, 1e-9))
        for i in OPT_PARAM_INDICES
    ]
    
    result = differential_evolution(
        func=lambda x: objective_function(
            x, base_params, model_params, experimental_data, variant
        ),
        bounds=bounds,
        strategy='best1bin',
        maxiter=200,
        popsize=15,
        tol=1e-7,
        updating='deferred',
        disp=True
    )

    optimized = base_params.copy()
    optimized[OPT_PARAM_INDICES] = result.x
    return optimized


def run_bayesian_optimization(base_params,
                               model_params,
                               experimental_data,
                               variant):
    """Optimize using Bayesian Optimization with Gaussian Processes.

    This method builds a probabilistic surrogate model (a Gaussian
    Process) of the objective function. It uses an acquisition function
    to intelligently select the next point to evaluate, balancing
    exploration of uncertain regions and exploitation of promising areas.
    It is highly effective for objective functions that are expensive
    to evaluate, which is common when simulations are involved.

    Args:
        base_params: np.ndarray, initial parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict, maps variants to data.
        variant: str, the variant to optimize for.

    Returns:
        np.ndarray, optimized parameter vector.
    """
    bounds = [
        (0, max(base_params[i] * 100, 1e-9))
        for i in OPT_PARAM_INDICES
    ]
    
    obj_func = lambda x: objective_function(
        np.array(x), base_params, model_params, experimental_data, variant
    )

    result = gp_minimize(
        func=obj_func,
        dimensions=bounds,
        n_calls=150,
        n_initial_points=15,
        acq_func="EI",
        random_state=123
    )

    optimized = base_params.copy()
    optimized[OPT_PARAM_INDICES] = result.x
    return optimized


def run_saea(base_params, model_params, experimental_data, variant):
    """Optimize using Surrogate-Assisted Evolutionary Algorithms (SAEA).

    SAEA uses a fast-to-evaluate surrogate model to approximate the
    expensive objective function, guiding the evolutionary search.
    Bayesian Optimization (implemented in the function above) is a prime
    example of a SAEA, where the surrogate is a Gaussian Process.
    This function serves as an alias for clarity and aligns with
    the terminology from your research paper.

    Args:
        base_params: np.ndarray, initial parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict, maps variants to data.
        variant: str, the variant to optimize for.

    Returns:
        np.ndarray, optimized parameter vector.
    """
    return run_bayesian_optimization(
        base_params, model_params, experimental_data, variant
    )


def run_pso(base_params, model_params, experimental_data, variant):
    """Optimize using Particle Swarm Optimization (PSO).

    PSO is a population-based algorithm where particles 'fly' through
    the parameter space. Each particle's movement is influenced by its
    own best-known position and the entire swarm's best-known position.
    It is effective for exploring large search spaces.

    Args:
        base_params: np.ndarray, initial parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict, maps variants to data.
        variant: str, the variant to optimize for.

    Returns:
        np.ndarray, optimized parameter vector.
    """
    bounds_low = np.array([0 for _ in OPT_PARAM_INDICES])
    bounds_high = np.array([
        max(base_params[i] * 100, 1e-9) for i in OPT_PARAM_INDICES
    ])
    bounds = (bounds_low, bounds_high)
    

    def pso_objective(x):
        n_particles = x.shape[0]
        costs = np.zeros(n_particles)
        for i in range(n_particles):
            costs[i] = objective_function(
                x[i], base_params, model_params, experimental_data, variant
            )
        return costs
    
    
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    optimizer = GlobalBestPSO(
        n_particles=20,
        dimensions=len(OPT_PARAM_INDICES),
        options=options,
        bounds=bounds
    )

    cost, pos = optimizer.optimize(pso_objective, iters=100)
    
    optimized = base_params.copy()
    optimized[OPT_PARAM_INDICES] = pos
    return optimized