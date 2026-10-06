"""Performance / limitations slide: each model's error on each day of the real window (where everyone struggled)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from style import BLUE_RAMP, DATA, footnote, save, title  # noqa: E402

w = pd.read_csv(DATA / "final_window.csv", index_col=0)
w.index = pd.to_datetime(w.index, utc=True).tz_convert("America/New_York").tz_localize(None)
rows = {"MOS (ECMWF + station)": "MOS (ECMWF + correction)",
        "Raw ECMWF + climatology after day 11": "Raw ECMWF",
        "Random forest (Lexie)": "Random forest (Lexie)",
        "Station-only linear (Burak)": "Station-only linear (Burak)",
        "Ridge + alternative data (Sam)": "Ridge + alt. data (Sam)",
        "KNN analogs (Sam)": "KNN analogs (Sam)",
        "Baseline: same hour, prior years": "Baseline: same hour, prior years"}
err = w[list(rows)].sub(w["actual"], axis=0).abs()
daily = err.groupby(w.index.normalize()).mean().T
daily.index = [rows[r] for r in daily.index]

cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd"] + BLUE_RAMP)
fig, ax = plt.subplots(figsize=(14, 6.4))
im = ax.imshow(daily.values, cmap=cmap, vmin=0, vmax=8, aspect="auto")
for i in range(daily.shape[0]):
    for j in range(daily.shape[1]):
        v = daily.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11.5, color="white" if v > 4.5 else "#0b0b0b")
ax.set_xticks(range(daily.shape[1]))
ax.set_xticklabels([d.strftime("%b\n%-d") for d in daily.columns], fontsize=12)
ax.set_yticks(range(daily.shape[0]))
ax.set_yticklabels(daily.index, fontsize=13)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, daily.shape[1]), minor=True)
ax.set_yticks(np.arange(-0.5, daily.shape[0]), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.set_label("Mean absolute error that day (°C)")
cb.outline.set_visible(False)
title(ax, "Error by day on the real window: where every model struggled",
      "The weather model wins the first week; the Sep 23–25 cold spell beat every model")
footnote(fig, "Sep 30 covers 12am–9am only (the real data ends at 9:51am). Darker = larger error.")
save(fig, "fig8_error_heatmap.png")
