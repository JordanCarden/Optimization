import os

import numpy as np
import pandas as pd

from simulate import simulate_variant_response
from noise_model import apply_composite_noise

GROUND_TRUTH_PARAMS = np.array([
    0.006964897545080543,
    0.49845081326380275,
    0.2078202745464548,
    0.2710187946598219,
    0.27005607351523053,
    0.00010000824037478934,
    0.0038153750071495494,
    0.0035213879373759585,
    0.005417172568602964,
    0.009585891462446375,
    0.00010422205930833477,
    16228.473238857363,
    9.365769266178052e-06,
    781261.5525853611,
    22942.76082863546,
    118.23071774262185,
    9.999707447302254e-06,
    2.4959885327086166e-06,
    1.9708285065082913,
    0.006755173105916858,
    972508.551350425,
])

MODEL_PARAMS = {
    "P_x": 1e-9,
    "P_y": 1e-9,
    "P_z": 1e-9,
    "IPTG": 0.1e-3,
}

SIM_DURATION_S = 420 * 60
SIM_STEP_S = 0.1
SAMPLE_STEP = 6000
VARIANT = "AAV"
OUTPUT_DIR = "data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


def generate_datasets() -> None:
    """Generate and save synthetic datasets.

    A clean trace is simulated using the ground truth parameters. Three
    independent datasets are then created, each containing three noisy
    replicates. The resulting CSV files are stored in ``OUTPUT_DIR``.

    Returns:
        None: This function is executed for its side effects only.
    """
    time_s = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sampled_s = time_s[::SAMPLE_STEP]
    time_min = sampled_s / 60

    clean_trace = simulate_variant_response(
        params=GROUND_TRUTH_PARAMS, model_params=MODEL_PARAMS, variant=VARIANT
    )

    num_datasets = 3
    replicates = 3

    for i in range(1, num_datasets + 1):
        data = {"time_min": time_min}
        for j in range(1, replicates + 1):
            noisy = apply_composite_noise(trace=clean_trace, time_array=time_min)
            data[f"replicate_{j}"] = noisy

        df = pd.DataFrame(data)
        path = os.path.join(OUTPUT_DIR, f"dataset_{i}.csv")
        df.to_csv(path, index=False)


if __name__ == "__main__":
    generate_datasets()
