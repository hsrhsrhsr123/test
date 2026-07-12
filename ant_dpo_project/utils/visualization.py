"""Plotting helpers with one consistent, colour-blind-safe style.

All figures in the project route through here so the report reads as a single
visual system. Palette is Okabe-Ito based (safe for the common colour-vision
deficiencies) and each experimental condition has a fixed colour.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# fixed colour assignments                                                     #
# --------------------------------------------------------------------------- #
PALETTE = {
    "reference": "#6B7280",   # neutral gray
    "dpo": "#94A3B8",
    "weight": "#0072B2",      # blue    – Stage 1
    "rank": "#E69F00",        # orange  – Stage 2
    "list": "#009E73",        # green   – Stage 3
    "direct": "#D55E00",      # vermillion
    "iterative": "#CC79A7",   # purple
    "chosen": "#009E73",
    "rejected": "#D55E00",
}
STAGE_ORDER = ["reference", "weight", "rank", "list"]
STAGE_LABELS = {
    "reference": "Reference (SFT)",
    "dpo": "Vanilla DPO",
    "weight": "Stage 1 · Weight-DPO",
    "rank": "Stage 2 · Rank-DPO",
    "list": "Stage 3 · List-DPO",
    "direct": "Direct DPO",
    "iterative": "Iterative DPO",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 1.0,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def color(name: str) -> str:
    return PALETTE.get(name, "#333333")


def label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def savefig(fig, path: str) -> str:
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# reusable primitives                                                          #
# --------------------------------------------------------------------------- #
def violin(ax, data_by_group: dict, ylabel: str, title: str):
    """Violin plot; ``data_by_group`` maps group-name -> 1-D array."""
    names = list(data_by_group.keys())
    data = [np.asarray(data_by_group[n]) for n in names]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(color(names[i]))
        body.set_alpha(0.55)
        body.set_edgecolor(color(names[i]))
    if "cmeans" in parts:
        parts["cmeans"].set_color("#111827")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([label(n) for n in names], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def line(ax, x, y, name: str, **kw):
    ax.plot(x, y, color=color(name), label=label(name), linewidth=2.2,
            marker="o", markersize=4, **kw)


__all__ = ["PALETTE", "STAGE_ORDER", "STAGE_LABELS", "apply_style", "color",
           "label", "savefig", "violin", "line"]
