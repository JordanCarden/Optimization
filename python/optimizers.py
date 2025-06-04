import numpy as np
from scipy.optimize import differential_evolution

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


def run_genetic_optimization(base_params,
                             model_params,
                             experimental_data,
                             variant):
    """Optimize parameters using SciPy’s differential_evolution.

    Args:
        base_params: np.ndarray, original parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict[str, np.ndarray], maps variants to data.
        variant: str, one of simulate.VARIANTS.

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
        popsize=200,
        maxiter=200,
        tol=1e-7,
        disp=True
    )
    optimized = base_params.copy()
    optimized[OPT_PARAM_INDICES] = result.x
    return optimized


def run_simulated_annealing(base_params,
                            model_params,
                            experimental_data,
                            variant,
                            n_iterations=1000,
                            initial_temp=1.0,
                            cooling_rate=0.995):
    """Optimize parameters using simulated annealing.

    Args:
        base_params: np.ndarray, original parameter vector.
        model_params: dict, base model parameters.
        experimental_data: dict[str, np.ndarray], maps variants to data.
        variant: str, one of simulate.VARIANTS.
        n_iterations: int, number of annealing steps.
        initial_temp: float, starting temperature.
        cooling_rate: float, factor to multiply temperature per iteration.

    Returns:
        np.ndarray, best‐found parameter vector (base_params with OPT_PARAM_INDICES replaced).
    """
    bounds = [
        (0, max(base_params[i] * 100, 1e-9))
        for i in OPT_PARAM_INDICES
    ]
    current = base_params.copy()
    current_vals = base_params[OPT_PARAM_INDICES].copy()
    best = current.copy()
    best_obj = objective_function(
        current_vals, base_params, model_params, experimental_data, variant
    )
    temp = initial_temp

    for _ in range(n_iterations):
        candidate_vals = current_vals.copy()
        for idx, param_index in enumerate(OPT_PARAM_INDICES):
            low, high = bounds[idx]
            perturb = np.random.normal(
                loc=0.0, scale=(high - low) * 0.1
            )
            new_val = candidate_vals[idx] + perturb
            candidate_vals[idx] = np.clip(new_val, low, high)

        candidate = base_params.copy()
        candidate[OPT_PARAM_INDICES] = candidate_vals

        obj = objective_function(
            candidate_vals,
            base_params,
            model_params,
            experimental_data,
            variant
        )
        delta = obj - best_obj

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current = candidate.copy()
            current_vals = candidate_vals.copy()
            if obj < best_obj:
                best = candidate.copy()
                best_obj = obj

        temp *= cooling_rate
        if temp < 1e-8:
            temp = 1e-8

    return best