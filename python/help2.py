#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready plot for Parameter Recovery (Excluding Own Dataset)
Style: EXACT match to Fig01_RMSE_Synthetic
Output: publication_plots/Fig05_ParamRecovery_ExcludeOwn.{pdf,png}
"""

from __future__ import annotations
import ast, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # used for boxplot/swarmplot; no theme reset

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "publication_plots")
SYN_CSV = os.path.join(PROJECT_ROOT, "synthetic_results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Style (identical to Fig01_RMSE_Synthetic) ----------------
MM_TO_INCH = 1.0 / 25.4
FIG_WIDTH_MM = 90
FIG_HEIGHT_MM = 70
DPI = 600

plt.rcParams.update({
    "figure.figsize": (FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------- Config (same names & colors) ----------------
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}
ALGORITHMS = list(DISPLAY_NAMES.keys())
CUSTOM_COLORS = {
    "cmaes":          "#FB5607",
    "basin_hopping":  "#FFBE0B",
    "lshade":         "#8338EC",
    "dual_annealing": "#FF006E",
    "pso":            "#3A86FF",
}
COLOR_MAP = CUSTOM_COLORS
DISPLAY_COLOR_MAP = {DISPLAY_NAMES[a]: COLOR_MAP[a] for a in ALGORITHMS}

# ---------------- Bounds & helpers (MASE) ----------------
LOWER_BOUNDS = [
    0.0005,0.0005,0.0005,0.0005,0.0005,
    0.0001,0.0001,0.00005,0.0001,0.0001,0.00005,
    100,1e-8,100,100,100,1e-8,1e-8,0.5,0.0001,100
]
UPPER_BOUNDS = [
    0.5,0.5,0.5,0.5,0.5,
    0.01,0.01,0.01,0.01,0.01,0.01,
    1_000_000,1e-5,1_000_000,1_000_000,1_000_000,1e-5,1e-5,5,0.01,1_000_000
]

def _parse_params(param_str: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(param_str), dtype=float)

def _load_ground_truth(category: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"{category}_best_params.csv")
    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)

def _mase(found: np.ndarray, truth: np.ndarray) -> float:
    lb = np.asarray(LOWER_BOUNDS, dtype=float)
    ub = np.asarray(UPPER_BOUNDS, dtype=float)
    scale = ub - lb
    found_scaled = (found - lb) / scale
    truth_scaled = (truth - lb) / scale
    return float(np.mean(np.abs(found_scaled - truth_scaled)))

def _get_mase_df() -> pd.DataFrame:
    df = pd.read_csv(SYN_CSV)
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]
    cats = sorted(df["category"].unique())
    truth = {c: _load_ground_truth(c) for c in cats}
    df["mase"] = df.apply(
        lambda r: _mase(_parse_params(r["params"]), truth[r["category"]]),
        axis=1,
    )
    return df

# ---------------- Plot ----------------
def plot_param_recovery_excluding_own():
    if not os.path.exists(SYN_CSV):
        raise FileNotFoundError(f"Missing synthetic results: {SYN_CSV}")

    df = _get_mase_df()
    df = df[df["optimizer"] != df["category"]].copy()
    if df.empty:
        raise RuntimeError("No rows after excluding own-dataset.")

    df["algorithm"] = df["optimizer"].map(DISPLAY_NAMES)

    # Order by median (ascending)
    medians = df.groupby("algorithm")["mase"].median().sort_values()
    sorted_algorithms = medians.index.tolist()

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH),
        constrained_layout=True
    )

    # Light-gray boxes (no fliers), then colored points — EXACT params
    sns.boxplot(
        ax=ax, data=df, x="algorithm", y="mase",
        order=sorted_algorithms, color="lightgray",
        width=0.30, fliersize=0
    )
    sns.swarmplot(
        ax=ax, data=df, x="algorithm", y="mase",
        order=sorted_algorithms, hue="algorithm",
        palette=DISPLAY_COLOR_MAP, size=1.5, linewidth=0
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("MASE")
    ax.set_title("Parameter Recovery (Excluding Own Dataset)")

    # X tick labels styled exactly like Fig01
    ax.set_xticks(range(len(sorted_algorithms)))
    ax.set_xticklabels(sorted_algorithms)
    for lab in ax.get_xticklabels():
        lab.set_rotation(32)
        lab.set_horizontalalignment("center")
        lab.set_verticalalignment("top")
        lab.set_rotation_mode("anchor")
    ax.tick_params(axis="x", pad=12)

    # Median annotations (same offset/font)
    for i, algo in enumerate(sorted_algorithms):
        m = float(medians[algo])
        ax.text(i + 0.15, m, f"{m:.3f}",
                ha="left", va="center", fontsize=8, color="black")

    # Tight side padding + headroom (same pattern)
    ax.set_xlim(-0.5, len(sorted_algorithms) - 0.5)
    ax.margins(x=0)
    ymax = float(df["mase"].max())
    ax.set_ylim(0, ymax * 1.15)

    out_pdf = os.path.join(OUTPUT_DIR, "Fig05_ParamRecovery_ExcludeOwn.pdf")
    out_png = os.path.join(OUTPUT_DIR, "Fig05_ParamRecovery_ExcludeOwn.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter recovery (exclude own) to:\n  {out_pdf}\n  {out_png}")

if __name__ == "__main__":
    plot_param_recovery_excluding_own()
