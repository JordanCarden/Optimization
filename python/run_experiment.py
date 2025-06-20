import argparse
from scipy.io import loadmat

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
        choices=['de', 'anneal'],
        default='de',
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

    if args.method == 'anneal':
        algorithm_name = 'Simulated Annealing'
        optimized_params = optimizers.run_simulated_annealing(
            base_params,
            model_params,
            experimental_data,
            args.variant,
            n_iterations=1000,
            initial_temp=1.0,
            cooling_rate=0.995
        )
    else:
        algorithm_name = 'Genetic Optimization'
        optimized_params = optimizers.run_genetic_optimization(
            base_params,
            model_params,
            experimental_data,
            args.variant
        )


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
        algorithm_name
    )


if __name__ == '__main__':
    main()