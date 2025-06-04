import numpy as np
import matplotlib.pyplot as plt

from simulate import VARIANTS, COLOR_MAP


def plot_results(simulated,
                 experimental_data,
                 variant,
                 algorithm):
    """Plot experimental data for all variants and simulated for chosen.

    Args:
        simulated: np.ndarray, simulated GFP data for variant.
        experimental_data: dict[str, np.ndarray], maps variants to data.
        variant: str, one of VARIANTS.
        algorithm_name: str, name of the optimization algorithm.

    Returns:
        None.
    """
    t = np.arange(len(simulated))
    plt.figure()
    for var in VARIANTS:
        plt.plot(
            t,
            experimental_data[var],
            '*' + COLOR_MAP[var],
            label=f'{var} exp'
        )
    plt.plot(
        t,
        simulated,
        '-' + COLOR_MAP[variant],
        label=f'{variant} sim'
    )
    plt.xlabel('Time points (10 min intervals)')
    plt.ylabel('GFP')
    plt.legend()
    plt.title(f'{variant} {algorithm}')
    plt.show()