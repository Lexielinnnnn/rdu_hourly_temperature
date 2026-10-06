"""Performance slide: every model's error on the real Sep 17-30 2026 window, ranked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from style import AQUA, BLUE, DATA, GREY, INK, ORANGE, footnote, save, title  # noqa: E402

m = pd.read_csv(DATA / "final_metrics_all.csv").sort_values("MAE", ascending=False)
label = {"Raw ECMWF + climatology after day 11": "Raw ECMWF (no correction)"}
group = {"MOS (ECMWF + station)": ("Uses the ECMWF forecast + our correction", BLUE),
         "Raw ECMWF + climatology after day 11": ("Raw weather model", ORANGE)}


def style_of(name):
    if name in group:
        return group[name]
    if name.startswith("Baseline"):
        return ("Simple baseline", GREY)
    return ("Learned from station / regional data only", AQUA)


fig, ax = plt.subplots(figsize=(13, 6.8))
names = [label.get(n, n) for n in m["model"]]
colors = [style_of(n)[1] for n in m["model"]]
bars = ax.barh(names, m["MAE"], color=colors, height=0.66, edgecolor="white", linewidth=2)
for b, v in zip(bars, m["MAE"]):
    ax.text(v + 0.04, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=13.5, color=INK)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0, m["MAE"].max() * 1.12)
ax.set_xlabel("Mean absolute error (°C), lower is better")
ax.grid(axis="y", visible=False)
seen = {}
for n in m["model"][::-1]:
    seen.setdefault(*style_of(n))
ax.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()], loc="upper center", bbox_to_anchor=(0.35, -0.13),
          ncol=2, fontsize=12.5)
title(ax, "Real-world test: Sep 17–30 2026, 322 hours",
      "Station data alone plateaus near 3 °C; correcting the ECMWF forecast gets to 2.59 °C")
footnote(fig, "Every forecast was produced and saved before the real temperatures were downloaded; nothing was re-tuned afterwards. "
              "Raw ECMWF uses the normal for the date after its run ends (day 11).")
save(fig, "fig2_model_ladder.png", bottom=0.05)
