"""Model slide: how much the final MOS model trusts ECMWF's forecast, by forecast day."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from style import BLUE, DATA, INK, LIGHT, MUTED, footnote, save, title  # noqa: E402

c = pd.read_csv(DATA / "mos_coefficients.csv").set_index("feature")["coef_original_units"]
days = list(range(1, 12))
w = [c[f"nwp_anom x day{d}"] for d in days]

fig, ax = plt.subplots(figsize=(12.5, 6.4))
ax.axvspan(11.5, 14.5, color=LIGHT, zorder=0)
ax.text(13, 0.5, "No ECMWF\ndata this far", ha="center", va="center", color=MUTED, fontsize=13)
ax.bar(days, w, color=BLUE, width=0.72, edgecolor="white", linewidth=2)
for d, v in zip(days, w):
    if d in (1, 6, 11):
        ax.text(d, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=13.5, color="white", fontweight="bold")
ax.axhline(1.0, color=MUTED, lw=1.2)
ax.text(14.4, 1.015, "1.0 = take ECMWF at face value", ha="right", va="bottom", color=MUTED, fontsize=12.5)
ax.set_xticks(range(1, 15))
ax.set_xlim(0.4, 14.5)
ax.set_ylim(0, 1.12)
ax.set_xlabel("Forecast day")
ax.set_ylabel("Weight on ECMWF's forecast")
ax.grid(axis="x", visible=False)
title(ax, "How much our model trusts the weather model, by forecast day",
      "Learned by ridge regression: near-full trust for 3 days, then it leans back toward the normal")
footnote(fig, "Coefficient on ECMWF's predicted departure from the normal temperature, in original units, from the final MOS "
              "model (trained on everything before Sep 17 2026).")
save(fig, "fig4_trust_by_lead_day.png")
