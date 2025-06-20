import argparse
import numpy as np
from scipy.io import loadmat
import optimizers
from simulate import simulate_variant_response, VARIANTS
from plotting import plot_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize parameters and plot results.'
    )
    parser.add_argument(
        'variant',
        choices=VARIANTS,
        nargs='?',
        default='AAV',
        help='Degradation tag variant.'
    )
    parser.add_argument(
        'method',
        choices=['bo', 'cmaes', 'lshade', 'pso', 'direct'],
        default='direct',
        help='Optimization method.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for stochastic optimizers.'
    )
    return parser.parse_args()


def main():
    """Load data, run chosen optimizer, and plot results."""
    args = parse_args()
    mat_data = loadmat('matlab/experimental_data.mat')
    experimental_data = {
        var: mat_data[var].flatten() for var in VARIANTS
    }

    model_params = {
        'P_x': 1e-9,
        'P_y': 1e-9,
        'P_z': 1e-9,
        'IPTG': 0.1e-3
    }
    base_params = loadmat('matlab/initial_params.mat')['p'].flatten()

    opt_param_indices = list(range(len(base_params)))

    bounds = [
        (p * 0.1, p * 10) if p > 0 else (-1e-6, 1e-6)
        for p in base_params
    ]
    for i, (low, high) in enumerate(bounds):
        if low > high:
            bounds[i] = (high, low)

    np.random.seed(args.seed)

    if args.method == 'bo':
        algorithm_name = 'Bayesian Optimization'
        optimized_params = optimizers.run_bayesian_optimization(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            bounds,
            opt_param_indices,
            args.seed,
        )
    elif args.method == 'cmaes':
        algorithm_name = 'CMA-ES'
        optimized_params = optimizers.run_cma_es(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            bounds,
            opt_param_indices,
            args.seed,
        )
    elif args.method == 'lshade':
        algorithm_name = 'L-SHADE'
        optimized_params = optimizers.run_lshade(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            bounds,
            opt_param_indices,
        )
    elif args.method == 'pso':
        algorithm_name = 'Particle Swarm Optimization'
        optimized_params = optimizers.run_pso(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            bounds,
            opt_param_indices,
        )
    elif args.method == 'direct':
        algorithm_name = 'DIRECT'
        optimized_params = optimizers.run_direct(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            bounds,
            opt_param_indices,
        )

    print(f"Optimized parameters using {algorithm_name}:")
    print(optimized_params)

    simulated = simulate_variant_response(
        optimized_params,
        model_params,
        args.variant
    )
    plot_results(
        simulated,
        experimental_data,
        args.variant,
        algorithm_name,
    )


if __name__ == '__main__':
    main()
