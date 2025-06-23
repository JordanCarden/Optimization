import os
import numpy as np
import pandas as pd
from simulate import simulate_variant_response
from noise_model import apply_composite_noise

# Ground truth and model parameters remain the same.
GROUND_TRUTH_PARAMS = np.array([
    4.70940318e00, 4.82725805e-01, 1.01149449e01, 8.45102567e00,
    3.73637489e01, 2.16021069e-01, 3.05100532e-01, 3.76967367e-02,
    1.29482278e-01, 4.27994896e-01, 5.60767076e-01, 1.03341863e04,
    2.63288502e-04, 5.75797013e03, 8.35920780e03, 1.01590610e04,
    1.19814795e-02, 7.83780130e-03, 3.61074026e-01, 1.60724775e-04,
    2.93045286e03,
])

MODEL_PARAMS = {
    'P_x': 1e-9,
    'P_y': 1e-9,
    'P_z': 1e-9,
    'IPTG': 0.1e-3,
}

# Simulation and file settings
SIM_DURATION_S = 420 * 60
SIM_STEP_S = 0.1
SAMPLE_STEP = 6000
VARIANT = 'AAV'
OUTPUT_DIR = 'data'

def generate_and_save_all_data():
    """
    Generates and saves the clean simulation trace, the ground truth parameters,
    and the noisy replicate datasets to the specified output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Check for correct parameter count
    if GROUND_TRUTH_PARAMS.size != 21:
        print("Error: 'GROUND_TRUTH_PARAMS' must contain 21 values.")
        return

    # Set up time arrays for the simulation
    time_s = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sampled_time_s = time_s[::SAMPLE_STEP]
    time_min = sampled_time_s / 60

    # Generate the single clean trace from the ground truth parameters
    clean_trace = simulate_variant_response(
        params=GROUND_TRUTH_PARAMS,
        model_params=MODEL_PARAMS,
        variant=VARIANT
    )

    # --- New Functionality: Save the clean trace ---
    clean_trace_df = pd.DataFrame({'time_min': time_min, 'gfp': clean_trace})
    clean_trace_path = os.path.join(OUTPUT_DIR, 'clean_trace.csv')
    clean_trace_df.to_csv(clean_trace_path, index=False)
    print(f"Clean trace saved to: {clean_trace_path}")

    # --- New Functionality: Save the ground truth parameters ---
    gt_params_df = pd.DataFrame(
        {'parameter_value': GROUND_TRUTH_PARAMS}
    )
    gt_params_path = os.path.join(OUTPUT_DIR, 'ground_truth_params.csv')
    gt_params_df.to_csv(gt_params_path, index_label='parameter_index')
    print(f"Ground truth parameters saved to: {gt_params_path}")

    # --- Original Functionality: Generate and save noisy datasets ---
    num_datasets = 3
    replicates_per_dataset = 3
    for i in range(1, num_datasets + 1):
        dataset_data = {'time_min': time_min}
        for j in range(1, replicates_per_dataset + 1):
            noisy_trace = apply_composite_noise(
                trace=clean_trace,
                time_array=time_min
            )
            dataset_data[f'replicate_{j}'] = noisy_trace

        df = pd.DataFrame(dataset_data)
        path = os.path.join(OUTPUT_DIR, f'dataset_{i}.csv')
        df.to_csv(path, index=False)
        print(f"Noisy dataset {i} saved to: {path}")

if __name__ == '__main__':
    generate_and_save_all_data()