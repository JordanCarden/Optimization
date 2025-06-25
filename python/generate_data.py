import os

import numpy as np
import pandas as pd

from simulate import simulate_variant_response
from noise_model import apply_composite_noise

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

SIM_DURATION_S = 420 * 60
SIM_STEP_S = 0.1
SAMPLE_STEP = 6000
VARIANT = 'AAV'
OUTPUT_DIR = 'data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


def generate_datasets():
    """Generate and save synthetic datasets according to the experimental plan.

    Generates a clean trace using ground truth parameters and creates three
    datasets, each comprised of three noisy replicates. Results are saved as
    CSV files in OUTPUT_DIR.

    Args:
        None

    Returns:
        None
    """
    time_s = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sampled_s = time_s[::SAMPLE_STEP]
    time_min = sampled_s / 60

    clean_trace = simulate_variant_response(
        params=GROUND_TRUTH_PARAMS,
        model_params=MODEL_PARAMS,
        variant=VARIANT
    )

    num_datasets = 3
    replicates = 3

    for i in range(1, num_datasets + 1):
        data = {'time_min': time_min}
        for j in range(1, replicates + 1):
            noisy = apply_composite_noise(
                trace=clean_trace,
                time_array=time_min
            )
            data[f'replicate_{j}'] = noisy

        df = pd.DataFrame(data)
        path = os.path.join(OUTPUT_DIR, f'dataset_{i}.csv')
        df.to_csv(path, index=False)


if __name__ == '__main__':
    generate_datasets()
