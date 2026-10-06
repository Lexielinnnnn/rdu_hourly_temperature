"""Hero slide: the frozen MOS forecast vs what actually happened at RDU, Sep 17-30 2026."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from style import BLUE, DATA, GREY, INK, LIGHT, MUTED, footnote, save, title  # noqa: E402

w = pd.read_csv(DATA / "final_window.csv", index_col=0)
w.index = pd.to_datetime(w.index, utc=True).tz_convert("America/New_York").tz_localize(None)
m = pd.read_csv(DATA / "final_metrics_all.csv").set_index("model")
ecmwf_end = w["Raw ECMWF"].last_valid_index()

fig, ax = plt.subplots(figsize=(14, 6.6))
ax.axvspan(ecmwf_end, w.index[-1], color=LIGHT, zorder=0)
ax.text(ecmwf_end + pd.Timedelta(hours=4), 33.6, "ECMWF run ends:\nmodel falls back\nto the normal", color=MUTED,
        fontsize=12.5, va="top")

ax.plot(w.index, w["Baseline: climatology (normal)"], color=GREY, lw=1.6, ls=(0, (4, 3)), label="Normal for the date (climatology)")
ax.plot(w.index, w["actual"], color=INK, lw=2.2, label="Actual RDU temperature")
ax.plot(w.index, w["MOS (ECMWF + station)"], color=BLUE, lw=2.2, label="Our forecast (MOS), issued Sep 17 12am")

ax.annotate("Heat wave Sep 17–21: highs of 33–34 °C,\nforecast ran ~1.5 °C cool", xy=(pd.Timestamp("2026-09-21 14:00"), 34.4),
            xytext=(pd.Timestamp("2026-09-22 06:00"), 37.6), fontsize=12.5, color=INK, va="top",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate("Cold spell: highs near 18 °C, ~8 °C below normal,\nmissed by ECMWF and every other model",
            xy=(pd.Timestamp("2026-09-24 03:00"), 15.6), xytext=(pd.Timestamp("2026-09-19 12:00"), 9.6),
            fontsize=12.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

ax.set_ylim(7, 38.5)
ax.set_ylabel("Temperature (°C)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12.5)
ax.set_xlim(w.index[0], w.index[-1])
title(ax, "Our forecast vs what actually happened at RDU, Sep 17–30 2026",
      f"MOS mean absolute error {m.loc['MOS (ECMWF + station)', 'MAE']:.2f} °C over 322 hours, best of all our models")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
footnote(fig, "Forecast frozen before the real temperatures were downloaded. Uses the ECMWF run started Sep 16 8am EDT "
              "(published that afternoon) plus RDU observations up to 11:51pm Sep 16.")
save(fig, "fig1_forecast_vs_actual.png")
