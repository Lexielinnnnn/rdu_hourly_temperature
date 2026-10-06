"""Problem / features slide: temperature = normal for the date + a departure.  The departure is the hard part."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from style import BLUE, DATA, GREY, INK, MUTED, RED, footnote, save, title  # noqa: E402

d = pd.read_csv(DATA / "normal_vs_actual_2026.csv", index_col=0)
d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York").tz_localize(None)
g = d[["observed", "normal"]].resample("D")
day = g.mean()[g.count()["observed"] >= 20]        # drop partial days (Sep 30 only has data to 9am)
cut = pd.Timestamp("2026-09-17")

fig, ax = plt.subplots(figsize=(14, 6.4))
ax.fill_between(day.index, day["normal"], day["observed"], where=day["observed"] >= day["normal"],
                color=RED, alpha=0.35, lw=0, interpolate=True, label="Warmer than normal")
ax.fill_between(day.index, day["normal"], day["observed"], where=day["observed"] < day["normal"],
                color=BLUE, alpha=0.35, lw=0, interpolate=True, label="Colder than normal")
ax.plot(day.index, day["normal"], color=GREY, lw=2.2, label="Normal for the date (our climatology regression)")
ax.plot(day.index, day["observed"], color=INK, lw=2.2, label="Observed daily mean at RDU")
ax.axvline(cut, color=MUTED, lw=1.2)
ax.text(cut - pd.Timedelta(hours=10), 31.4, "Forecast issued\n(12am Sep 17)", color=INK, fontsize=12.5, va="top", ha="right")
ax.text(pd.Timestamp("2026-09-23 12:00"), 15.4, "Cold spell", color=INK, fontsize=12.5, ha="center", va="top")
ax.text(pd.Timestamp("2026-09-19 12:00"), 28.0, "Heat wave", color=INK, fontsize=12.5, ha="center", va="bottom")
ax.set_ylim(13, 32)
ax.set_ylabel("Daily mean temperature (°C)")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.TH))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
ax.set_xlim(day.index[0], day.index[-1])
title(ax, "Temperature = normal for the date + a departure",
      "The normal is easy to model; the shaded departure is what every model is really trying to predict")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12.5)
footnote(fig, "Normal = Fourier regression on day of year x hour of day, fit only on data before Sep 17 2026. "
              "Observations after Sep 17 are shown for illustration; no model saw them.")
save(fig, "fig7_normal_vs_actual.png", bottom=0.05)
