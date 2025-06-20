# python/run_experiment.py

import argparse
from scipy.io import loadmat
import numpy as np

import optimizers
from simulate import simulate_variant_response, VARIANTS
from plotting import plot_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize parameters and plot results.')
    parser.add_argument(
        'variant',
        choices=VARIANTS,
        nargs='?',
        default='AAV',
        help='Degradation tag variant.'
    )
    parser.add_argument(
        'method',
        choices=['cma', 'lshade', 'bayes', 'saea', 'pso'],
        default='cma',
        help='Optimization method.'
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

    optimizers.OPT_PARAM_INDICES = list(range(len(base_params)))
    

    optimizer_map = {
        'cma': ('CMA-ES', optimizers.run_cma_es),
        'lshade': ('L-SHADE', optimizers.run_lshade),
        'bayes': ('Bayesian Optimization', optimizers.run_bayesian_optimization),
        'saea': ('Surrogate-Assisted EA', optimizers.run_saea),
        'pso': ('Particle Swarm Opt (PSO)', optimizers.run_pso)
    }

    if args.method in optimizer_map:
        algorithm_name, optimizer_func = optimizer_map[args.method]
        optimized_params = optimizer_func(
            base_params,
            model_params,
            experimental_data,
            args.variant
        )
    else:
        raise ValueError(f"Unknown optimization method: {args.method}")


    print(f"\nOptimized parameters from {algorithm_name}:\n{optimized_params}")

    simulated = simulate_variant_response(
        optimized_params,
        model_params,
        args.variant
    )
    plot_results(
        simulated,
        experimental_data,
        args.variant,
        algorithm_name
    )


if __name__ == '__main__':
    main()