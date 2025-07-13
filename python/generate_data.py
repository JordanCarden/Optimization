"""Generate noisy synthetic datasets from optimized parameter sets."""

import os

import numpy as np
import pandas as pd

from noise_model import apply_composite_noise
from simulate import simulate_variant_response

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
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")


def _load_parameters(path: str) -> np.ndarray:
    """Load optimized parameters from a CSV file."""

    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def generate_datasets() -> None:
    """Generate and save synthetic datasets.

    Parameters for five different optimizers are loaded from the ``data``
    directory. For each optimizer, three independent noisy datasets are created.
    Each dataset contains three replicates. Output files are named
    ``<optimizer>_<index>.csv`` and stored in ``data``.

    Returns:
        None: This function is executed for its side effects only.
    """
    time_s = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sampled_s = time_s[::SAMPLE_STEP]
    time_min = sampled_s / 60

    param_files = {
        "basin_hopping": "basin_hopping_best_params.csv",
        "cmaes": "cmaes_best_params.csv",
        "dual_annealing": "dual_annealing_best_params.csv",
        "lshade": "lshade_best_params.csv",
        "pso": "pso_best_params.csv",
    }

    replicates = 3

    for optimizer, file_name in param_files.items():
        params_path = os.path.join(DATA_DIR, file_name)
        params = _load_parameters(params_path)
        clean_trace = simulate_variant_response(
            params=params, model_params=MODEL_PARAMS, variant=VARIANT
        )

        for idx in range(1, 4):
            data = {"time_min": time_min}
            for rep in range(1, replicates + 1):
                noisy = apply_composite_noise(
                    trace=clean_trace, time_array=time_min
                )
                data[f"replicate_{rep}"] = noisy

            df = pd.DataFrame(data)
            path = os.path.join(DATA_DIR, f"{optimizer}_{idx}.csv")
            df.to_csv(path, index=False)


if __name__ == "__main__":
    generate_datasets()
