import numpy as np
from scipy.integrate import solve_ivp

from protein_model import compute_model_derivatives

ATC_CONVERSION = 0.46822
SIM_DURATION_S = 420 * 60
SIM_STEP_S = 0.1
SAMPLE_STEP = 6000
VARIANTS = ['AAV', 'ASV', 'LVA', 'noTetR']
COLOR_MAP = {
    'AAV': 'r',
    'ASV': 'b',
    'LVA': 'k',
    'noTetR': 'm'
}


def simulate_variant_response(params, model_params, variant):
    """Simulate GFP response for a given degradation tag variant.

    Args:
        params: np.ndarray, fitted parameter vector.
        model_params: dict, base model parameters with keys 'P_x', 'P_y', 'P_z'.
        variant: str, one of VARIANTS.

    Returns:
        np.ndarray of GFP at sampled time points.
    """
    sim_params = model_params.copy()
    sim_params['aTc'] = (50 / ATC_CONVERSION) * 1e-9
    state0 = np.zeros(11)
    state0[3] = sim_params['aTc']

    params_mod = params.copy()
    if variant == 'AAV':
        params_mod[7] *= 2
    elif variant == 'LVA':
        params_mod[7] *= 12
    elif variant == 'noTetR':
        params_mod[1] = 0
        params_mod[7] = 0

    t_eval = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sol = solve_ivp(
        lambda t, y: compute_model_derivatives(
            t, y, params_mod, sim_params),
        [t_eval[0], t_eval[-1]],
        state0,
        t_eval=t_eval,
        method='BDF',
        atol=1e-11,
        rtol=1e-11
    )
    concentrations_nanomolar = sol.y.T * 1e9
    return concentrations_nanomolar[::SAMPLE_STEP, 10] * 10**params_mod[18]