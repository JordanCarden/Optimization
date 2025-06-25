import numpy as np


def apply_composite_noise(
    trace: np.ndarray,
    time_array: np.ndarray,
    a: float = 25.0,
    b: float = 1.0,
    rho: float = 0.007,
    sigma_delta: float = 0.1,
    drift_slope: float = 0.01
):
    """Apply composite noise model to a simulated trace.

    Default parameters are based on the Composite Noise Model Report.

    Args:
        trace: Clean simulation output.
        time_array: Time axis in minutes.
        a: Baseline noise variance.
        b: Scale factor for signal-dependent noise.
        rho: AR(1) correlation coefficient for temporal stickiness.
        sigma_delta: Multiplicative noise scale for percent-level errors.
        drift_slope: Linear drift per minute.

    Returns:
        Synthetic trace with composite noise applied.
    """
    length = trace.size
    e_autocorr = np.zeros(length)
    noisy = np.zeros(length)

    for t in range(length):
        var = a + b * trace[t]
        eta = np.random.normal(scale=np.sqrt(max(0, var)))

        if t == 0:
            e_autocorr[t] = eta
        else:
            e_autocorr[t] = rho * e_autocorr[t - 1] + eta

        delta = np.random.normal(scale=sigma_delta)
        drift = drift_slope * time_array[t]
        noisy[t] = trace[t] * (1 + delta) + e_autocorr[t] + drift

    return noisy
