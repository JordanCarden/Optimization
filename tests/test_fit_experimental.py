import os
import sys

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, TEST_DIR)  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fit_experimental import _load_aav_trace  # noqa: E402
from objective_function import ObjectiveTracker  # noqa: E402


def test_load_aav_trace() -> None:
    data = _load_aav_trace("data/experimental_data.mat")
    assert data.shape == (43,)


def test_objective_evaluate() -> None:
    trace = _load_aav_trace("data/experimental_data.mat")
    base_params = pd.read_csv("data/ground_truth_params.csv")["parameter_value"].values
    tracker = ObjectiveTracker(
        dataset_path=None,
        base_params=base_params,
        opt_param_indices=list(range(len(base_params))),
        mean_trace=trace,
    )
    sse = tracker.evaluate(base_params)
    assert np.isfinite(sse)
