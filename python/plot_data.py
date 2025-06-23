import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

DATA_DIR = 'data'
MAT_FILE = 'experimental_data.mat'
NUM_DATASETS = 3
REPLICATE_COLS = ['replicate_1', 'replicate_2', 'replicate_3']


def plot_with_experimental_data():
    """Load and plot synthetic datasets with AAV experimental data overlay."""
    print("--- Starting Plot Script ---")

    mat_path = os.path.join(DATA_DIR, MAT_FILE)
    if not os.path.exists(mat_path):
        print(
            "[Error] The file "
            f"'{MAT_FILE}' was not found in '{DATA_DIR}' directory."
        )
        return

    print(f"Loading experimental data from '{mat_path}'...")
    mat_data = loadmat(mat_path)

    if 'AAV' not in mat_data:
        print(f"[Error] Could not find 'AAV' key in '{MAT_FILE}'.")
        return

    aav_exp_trace = mat_data['AAV'].ravel()
    print("Experimental data loaded successfully.")

    plots_created = 0
    for i in range(1, NUM_DATASETS + 1):
        file_path = os.path.join(DATA_DIR, f'dataset_{i}.csv')
        if not os.path.exists(file_path):
            print(
                "[Warning] Synthetic data file not found at "
                f"'{file_path}'. Skipping."
            )
            continue

        print(f"Plotting synthetic dataset {i} with experimental data...")
        df = pd.read_csv(file_path)
        time_axis = df['time_min']

        if len(time_axis) != len(aav_exp_trace):
            print(
                "[Warning] Mismatch in data points between synthetic "
                f"({len(time_axis)}) and experimental "
                f"({len(aav_exp_trace)}) data."
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        mean_trace = df[REPLICATE_COLS].mean(axis=1)
        std_trace = df[REPLICATE_COLS].std(axis=1)

        for col in REPLICATE_COLS:
            ax.plot(
                time_axis,
                df[col],
                color='gray',
                alpha=0.3,
                label='Synthetic Replicate'
            )

        ax.plot(
            time_axis,
            mean_trace,
            color='blue',
            linewidth=2.5,
            label='Mean of Synthetic Data'
        )
        ax.fill_between(
            time_axis,
            mean_trace - std_trace,
            mean_trace + std_trace,
            color='blue',
            alpha=0.2
        )
        ax.plot(
            time_axis,
            aav_exp_trace,
            linestyle='None',
            marker='o',
            markersize=5,
            color='red',
            label='AAV Experimental Data'
        )

        ax.set_title(
            f"Synthetic Dataset {i} vs. AAV Experimental Data", fontsize=16
        )
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('GFP (AU)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        plt.tight_layout()
        plots_created += 1

    if plots_created:
        print(f"\nDisplaying {plots_created} plot(s)...")
        plt.show()
    else:
        print("\nNo plots were created; synthetic files not found.")


if __name__ == '__main__':
    plot_with_experimental_data()
