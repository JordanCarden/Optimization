import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from simulate import (
    simulate_variant_response,
    VARIANTS,
    COLOR_MAP,
    SIM_STEP_S,
    SAMPLE_STEP,
)


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-type',
                        choices=['gaussian', 'composite'],
                        default='composite')
    parser.add_argument('--noise-std', type=float, default=25.0)
    parser.add_argument('--a', type=float, default=25.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.007)
    parser.add_argument('--sigma-delta', type=float, default=0.05)
    parser.add_argument('--drift-slope', type=float, default=0.01)
    parser.add_argument('--num-samples', type=int, default=10)
    return parser.parse_args()


def apply_gaussian_noise(trace, noise_std):
    """Add zero-mean Gaussian noise to a time-series.

    Args:
        trace (np.ndarray): Clean simulation output.
        noise_std (float): Standard deviation of noise.

    Returns:
        np.ndarray: Noisy trace.
    """
    return trace + np.random.normal(scale=noise_std, size=trace.shape)


def apply_composite_noise(trace, time, a, b, rho,
                          sigma_delta, drift_slope):
    """Apply composite noise model to a simulated trace.

    Args:
        trace (np.ndarray): Clean simulation output.
        time (np.ndarray): Time axis in minutes.
        a (float): Baseline noise variance.
        b (float): Scale factor for variance.
        rho (float): AR(1) correlation coefficient.
        sigma_delta (float): Multiplicative noise scale.
        drift_slope (float): Linear drift per minute.

    Returns:
        np.ndarray: Synthetic trace with noise.
    """
    length = trace.size
    e_autocorr = np.zeros(length)
    noisy = np.zeros(length)
    for t in range(length):
        var = a + b * trace[t]
        eta = np.random.normal(scale=np.sqrt(max(0, var)))
        e_autocorr[t] = (
            eta if t == 0 else rho * e_autocorr[t - 1] + eta
        )
        delta = np.random.normal(scale=sigma_delta)
        drift = drift_slope * time[t]
        noisy[t] = trace[t] * (1 + delta) + e_autocorr[t] + drift
    return noisy


def visualize_aav_with_experiments(true_params_list, a, b, rho,
                                  sigma_delta, drift_slope,
                                  num_samples, noise_type,
                                  noise_std):
    """Plot noisy synthetic AAV data and experiments for all variants.

    Args:
        true_params_list (Sequence[float]): Parameter vector.
        a (float): Baseline noise variance.
        b (float): Scale factor for variance.
        rho (float): AR(1) coefficient.
        sigma_delta (float): Multiplicative noise scale.
        drift_slope (float): Linear drift per minute.
        num_samples (int): Synthetic sample count.
        noise_type (str): 'gaussian' or 'composite'.
        noise_std (float): Std dev for Gaussian noise.
    """
    true_params = np.array(true_params_list)
    mat = loadmat('matlab/experimental_data.mat')
    model_params = {
        'P_x': 1e-9,
        'P_y': 1e-9,
        'P_z': 1e-9,
        'IPTG': 0.1e-3,
    }
    clean0 = simulate_variant_response(true_params, model_params, 'AAV')
    length = clean0.size
    time = np.arange(length) * SIM_STEP_S * SAMPLE_STEP / 60
    plt.figure()
    for variant in VARIANTS:
        exp_trace = mat[variant].ravel()
        color = COLOR_MAP.get(variant, 'gray')
        plt.plot(time,
                 exp_trace,
                 marker='o',
                 linestyle='None',
                 label=f'{variant} experimental',
                 color=color)
    for i in range(num_samples):
        if noise_type == 'gaussian':
            noisy = apply_gaussian_noise(clean0, noise_std)
        else:
            noisy = apply_composite_noise(clean0,
                                          time,
                                          a,
                                          b,
                                          rho,
                                          sigma_delta,
                                          drift_slope)
        plt.plot(time, noisy, label=f'{noise_type} synthetic {i+1}')
    plt.xlabel('Time (min)')
    plt.ylabel('GFP (AU)')
    plt.title('AAV Synthetic with Noise and Experiments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Parse arguments and run the visualization."""
    args = parse_args()
    user_true_params = [
        4.70940318e00, 4.82725805e-01, 1.01149449e01, 8.45102567e00,
        3.73637489e01, 2.16021069e-01, 3.05100532e-01, 3.76967367e-02,
        1.29482278e-01, 4.27994896e-01, 5.60767076e-01, 1.03341863e04,
        2.63288502e-04, 5.75797013e03, 8.35920780e03, 1.01590610e04,
        1.19814795e-02, 7.83780130e-03, 3.61074026e-01, 1.60724775e-04,
        2.93045286e03,
    ]
    visualize_aav_with_experiments(
        user_true_params,
        args.a,
        args.b,
        args.rho,
        args.sigma_delta,
        args.drift_slope,
        args.num_samples,
        args.noise_type,
        args.noise_std,
    )


if __name__ == '__main__':
    main()