"""Evaluation slide: backtest error by forecast day for MOS, raw ECMWF, station-only linear and climatology."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from style import DATA, LIGHT, MODEL_COLOR, MUTED, footnote, save, title  # noqa: E402

bt = pd.read_csv(DATA / "mos_backtest.csv")
g = bt.groupby(["model", "lead_day"])[["abs_err", "n"]].sum()
by_day = (g.abs_err / g.n).unstack(0)
n_fc = bt["origin"].nunique()

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.axvspan(11.5, 14.5, color=LIGHT, zorder=0)
ax.text(11.65, 1.08, "Beyond the ECMWF run:\neveryone uses the normal", color=MUTED, fontsize=12.5, va="bottom")
order = ["Climatology", "Station-only linear", "Raw ECMWF", "MOS (ECMWF + station)"]
labels = {"Climatology": "Climatology (normal for the date)", "Station-only linear": "Station-only linear (Burak)",
          "Raw ECMWF": "Raw ECMWF", "MOS (ECMWF + station)": "MOS: ECMWF + our correction"}
for name in order:
    s = by_day[name]
    ax.plot(s.index, s.values, "-o", color=MODEL_COLOR[name], ms=8, lw=2.2, label=labels[name],
            mec="white", mew=1.5, zorder=3 if name.startswith("MOS") else 2)
for x, y, txt, va in [(6.1, 1.80, "MOS: ECMWF + our correction", "top"),       # direct labels in empty space
                      (3.1, 1.95, "Raw ECMWF", "bottom"),
                      (2.15, 2.42, "Station-only linear", "center"),
                      (1.05, 3.02, "Climatology", "bottom")]:
    ax.text(x, y, txt, color="#0b0b0b", fontsize=12.5, va=va, ha="left")
ax.set_xticks(range(1, 15))
ax.set_xlim(0.6, 14.5)
ax.set_ylim(1.0, 3.6)
ax.set_xlabel("Forecast day (day 1 = Sep 17 in the real forecast)")
ax.set_ylabel("Mean absolute error (°C)")
title(ax, f"Backtest: error by forecast day ({n_fc} past forecasts, Aug 15 – Oct 15, 2024–25)",
      "MOS beats raw ECMWF on every day the run covers; station data alone only helps for ~2 days")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12.5)
footnote(fig, "Forecasts issued every 3 days at local midnight. For each season the model is retrained only on data "
              "before that season's first forecast.")
save(fig, "fig3_lead_day_skill.png")
