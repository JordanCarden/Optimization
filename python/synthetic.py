import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Assuming simulate.py, protein_model.py are in the same directory
# or accessible in the Python path.
# COLOR_MAP is imported from simulate.py as it's a shared resource.
from simulate import (
    simulate_variant_response,
    VARIANTS,
    COLOR_MAP,
    SIM_DURATION_S,
    SIM_STEP_S,
    SAMPLE_STEP
)

def visualize_synthetic_data_standalone(true_params_list):
    """
    Generates and visualizes synthetic data using the provided true parameters,
    and plots it against the experimental data.

    Args:
        true_params_list: list, the 21 "true" parameters for the model.
    """
    true_params = np.array(true_params_list)

    # These model_params are standard for your simulations
    model_params = {
        'P_x': 1e-9,
        'P_y': 1e-9,
        'P_z': 1e-9,
        'IPTG': 0.1e-3
    }

    # Load experimental data
    mat_data = loadmat('matlab/experimental_data.mat')
    experimental_data = {
        var: mat_data[var].flatten() for var in VARIANTS
    }

    synthetic_gfp_all_variants = {}

    print("Generating synthetic data for visualization...")
    for variant in VARIANTS:
        print(f"Simulating: {variant}")
        simulated_gfp = simulate_variant_response(
            true_params,
            model_params,
            variant
        )
        synthetic_gfp_all_variants[variant] = simulated_gfp
        print(f"Finished: {variant}")

    # Create a time column for plotting
    # Number of time points:
    num_points = int((SIM_DURATION_S / SIM_STEP_S) / SAMPLE_STEP) + 1
    # Time interval in minutes:
    time_interval_min = (SIM_STEP_S * SAMPLE_STEP) / 60
    time_points = np.arange(num_points) * time_interval_min

    # Plotting directly using matplotlib.pyplot
    plt.figure(figsize=(10, 6))
    for variant_name in VARIANTS:
        # Plot synthetic data
        if variant_name in synthetic_gfp_all_variants:
            plt.plot(
                time_points,
                synthetic_gfp_all_variants[variant_name],
                linestyle='-',
                color=COLOR_MAP.get(variant_name, 'gray'),
                label=f'{variant_name} Synthetic'
            )
        # Plot experimental data
        if variant_name in experimental_data:
            plt.plot(
                time_points,
                experimental_data[variant_name],
                marker='o',
                linestyle='None',
                color=COLOR_MAP.get(variant_name, 'gray'),
                label=f'{variant_name} Experimental'
            )

    plt.xlabel('Time (min)')
    plt.ylabel('Simulated and Experimental GFP Concentration (nM)')
    plt.title('Synthetic vs. Experimental GFP Response for All Variants')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()


if __name__ == '__main__':
    # The "true" parameters you provided
    user_true_parameters = [
        4.70940318e+00, 4.82725805e-01, 1.01149449e+01, 8.45102567e+00,
        3.73637489e+01, 2.16021069e-01, 3.05100532e-01, 3.76967367e-02,
        1.29482278e-01, 4.27994896e-01, 5.60767076e-01, 1.03341863e+04,
        2.63288502e-04, 5.75797013e+03, 8.35920780e+03, 1.01590610e+04,
        1.19814795e-02, 7.83780130e-03, 3.61074026e-01, 1.60724775e-04,
        2.93045286e+03
    ]

    visualize_synthetic_data_standalone(user_true_parameters)