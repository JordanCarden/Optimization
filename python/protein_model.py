import numpy as np


def compute_model_derivatives(time, state, params, model_params):
    """Compute derivative for protein detailed ODE model.

    Args:
        time: float, simulation time.
        state: np.ndarray with length 11, species concentrations.
        params: np.ndarray, fitted parameter vector.
        model_params: dict, base parameters with keys 'P_x', 'P_y', 'P_z'.

    Returns:
        np.ndarray with length 11, derivatives d(state)/dt.
    """
    derivatives = np.zeros(11)
    (STAR, THS, TetR, aTc, tetR_aTc, Y,
     Y_active, Pz_repressed, Pz_active, Z,
     GFP) = state
    free_Pz = model_params['P_z'] - Pz_repressed - Pz_active

    derivatives[0] = (
        model_params['P_x'] * params[0]
        - params[11] * STAR * free_Pz
        - params[5] * STAR
        + params[12] * Pz_active
    )
    derivatives[1] = (
        model_params['P_x'] * params[0]
        - params[13] * THS * Y
        - params[6] * THS
    )
    derivatives[2] = (
        params[1] * Y_active
        - params[14] * TetR * free_Pz
        - params[15] * TetR * aTc
        - params[7] * TetR
        + params[16] * Pz_repressed
        + params[17] * tetR_aTc
        - params[20] * TetR * Pz_active
    )
    derivatives[3] = -params[15] * TetR * aTc + params[17] * tetR_aTc
    derivatives[4] = params[15] * TetR * aTc - params[17] * tetR_aTc
    derivatives[5] = (
        params[2] * model_params['P_y']
        - params[13] * THS * Y
        - params[8] * Y
    )
    derivatives[6] = params[13] * THS * Y - params[19] * Y_active
    derivatives[7] = (
        params[14] * TetR * free_Pz
        - params[16] * Pz_repressed
        + params[20] * TetR * Pz_active
    )
    derivatives[8] = (
        params[11] * STAR * free_Pz
        - params[12] * Pz_active
        - params[20] * TetR * Pz_active
    )
    derivatives[9] = params[3] * Pz_active - params[9] * Z
    derivatives[10] = params[4] * Z - params[10] * GFP

    return derivatives
