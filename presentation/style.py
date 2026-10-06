"""Shared look for the presentation figures.

Same validated palette as final_model/plot_style.py (categorical slots 1-3 + neutrals) so the team's figures match.
Colors follow the MODEL, never its rank: a model has the same color on every slide.
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
RED = "#e34948"                                                # diverging warm pole (blue <-> red)
INK, MUTED, GRID, GREY, LIGHT = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8983", "#f0efec"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential

MODEL_COLOR = {                       # one color per model, used on every slide
    "MOS (ECMWF + station)": BLUE,
    "Raw ECMWF": ORANGE,
    "Station-only linear": AQUA,
    "Climatology": GREY,
    "Actual": INK,
}

DATA = Path("presentation/data")
OUT = Path("presentation/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold", "axes.labelsize": 15,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "font.family": "sans-serif",
    "lines.linewidth": 2.0,
})


def title(ax, text, sub=None):
    """Bold title plus an optional one-line takeaway underneath, both left-aligned."""
    ax.set_title(text, loc="left", pad=34 if sub else 12)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=14, ha="left", va="bottom")


def footnote(fig, text):
    fig.text(0.01, 0.01, textwrap.fill(text, 140), color=MUTED, fontsize=11.5, ha="left", va="bottom")


def save(fig, name, bottom=0.06):
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)
