"""Plot simulated trace against experimental data trace."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd

from generate_data import MODEL_PARAMS
from simulate import simulate_variant_response


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Configured arguments.
    """
    parser = argparse.ArgumentParser(description="Plot simulated trace.")
    parser.add_argument(
        "--param-file",
        type=str,
        default=os.path.join("data", "ground_truth_params.csv"),
        help="CSV file containing parameter values under 'parameter_value'",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="AAV",
        choices=["AAV", "ASV", "LVA", "noTetR"],
        help="Variant label to simulate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure instead of displaying it",
    )
    return parser.parse_args()


def load_params(path: str) -> np.ndarray:
    """Load parameter values from a CSV file."""
    df = pd.read_csv(path)
    if "parameter_value" not in df.columns:
        raise ValueError("CSV file must contain 'parameter_value' column")
    return df["parameter_value"].values.astype(float)


def load_experimental_trace(variant: str) -> np.ndarray:
    """Load experimental trace for a given variant from MAT file."""
    mat_path = os.path.join("data", "experimental_data.mat")
    mat_data = loadmat(mat_path)
    if variant not in mat_data:
        raise KeyError(f"Variant '{variant}' not found in {mat_path}")
    return mat_data[variant].ravel()


def build_time_axis(num_points: int) -> np.ndarray:
    """Construct time axis assuming 10 minute sampling."""
    return np.arange(num_points) * 10


def plot_traces(
    time: np.ndarray, sim: np.ndarray, exp: np.ndarray, output: str | None
) -> None:
    """Plot simulated and experimental traces."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, exp, "o", color="red", label="Experimental")
    plt.plot(time, sim, "-", color="blue", label="Simulated")
    plt.xlabel("Time (min)")
    plt.ylabel("GFP (AU)")
    plt.title("Simulated vs Experimental Trace")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if output:
        plt.savefig(output)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def main() -> None:
    """Run the plotting routine."""
    args = parse_args()
    params = load_params(args.param_file)
    sim_trace = simulate_variant_response(params, MODEL_PARAMS, args.variant)
    exp_trace = load_experimental_trace(args.variant)
    if sim_trace.size != exp_trace.size:
        raise ValueError("Simulated and experimental traces must be same length")
    time_axis = build_time_axis(sim_trace.size)
    plot_traces(time_axis, sim_trace, exp_trace, args.output)


if __name__ == "__main__":
    main()
