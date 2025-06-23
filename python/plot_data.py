import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

DATA_DIR = 'data'
MAT_FILE = 'experimental_data.mat'
NUM_DATASETS = 3
REPLICATE_COLS = [
    'replicate_1',
    'replicate_2',
    'replicate_3'
]


def plot_with_experimental_data():
    """Load and plot synthetic datasets with AAV experimental data."""
    mat_path = os.path.join(DATA_DIR, MAT_FILE)
    mat_data = loadmat(mat_path)
    aav_exp_trace = mat_data['AAV'].ravel()

    for i in range(1, NUM_DATASETS + 1):
        file_path = os.path.join(
            DATA_DIR,
            f'dataset_{i}.csv'
        )
        df = pd.read_csv(file_path)
        time_axis = df['time_min']

        mean_trace = df[REPLICATE_COLS].mean(axis=1)
        std_trace = df[REPLICATE_COLS].std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

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
            f"Synthetic Dataset {i} vs. AAV Experimental Data",
            fontsize=16
        )
        ax.set_xlabel(
            'Time (min)',
            fontsize=12
        )
        ax.set_ylabel(
            'GFP (AU)',
            fontsize=12
        )
        ax.grid(
            True,
            which='both',
            linestyle='--',
            linewidth=0.5
        )

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys()
        )

        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    plot_with_experimental_data()