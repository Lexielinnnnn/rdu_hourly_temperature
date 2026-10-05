"""Shared plot style for all slide figures (palette: validated categorical slots 1-3 + neutrals)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e3e2dd", "#8a8983"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 14,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
                     "font.family": "sans-serif"})
OUT = Path("figures"); OUT.mkdir(exist_ok=True)


def legend_below(ax, ncol=2, offset=-0.17):
    """Legend under the x-axis label (never over it)."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol)


def footnote(fig, text):
    import textwrap
    fig.text(0.01, 0.008, textwrap.fill(text, 105), color=MUTED, fontsize=11, ha="left", va="bottom")


def save(fig, name):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)
