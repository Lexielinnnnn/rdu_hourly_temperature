"""Performance slide: R^2 on the real Sep 17-30 2026 window, and how much of it is beyond the normal for the date."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from style import AQUA, BLUE, DATA, GREY, INK, MUTED, ORANGE, OUT, footnote  # noqa: E402

win = pd.read_csv(DATA / "final_window.csv", index_col=0)
win = win[win["actual"].notna()]
y = win["actual"].to_numpy()
models = [c for c in win.columns if c not in ("actual", "Raw ECMWF")]          # same models as the performance ladder
sse = {m: float(np.sum((win[m].to_numpy() - y) ** 2)) for m in models}
sst = float(np.sum((y - y.mean()) ** 2))
normal = "Baseline: climatology (normal)"
r2 = {m: 1 - sse[m] / sst for m in models}                                    # share of the temperature variation explained
skill = {m: 1 - sse[m] / sse[normal] for m in models}                          # share of the normal's error removed

label = {"Raw ECMWF + climatology after day 11": "Raw ECMWF (no correction)"}
group = {"MOS (ECMWF + station)": ("Uses the ECMWF forecast + our correction", BLUE),
         "Raw ECMWF + climatology after day 11": ("Raw weather model", ORANGE)}


def style_of(name):
    if name in group:
        return group[name]
    if name.startswith("Baseline"):
        return ("Simple baseline", GREY)
    return ("Learned from station / regional data only", AQUA)


order = sorted(models, key=lambda m: r2[m])                                    # worst at the bottom
names = [label.get(m, m) for m in order]
colors = [style_of(m)[1] for m in order]

fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.8), sharey=True)
fig.subplots_adjust(left=0.25, right=0.985, top=0.80, bottom=0.30, wspace=0.07)
N = len(order)
top = N - 0.15                                       # headroom above the first bar for the in-panel notes

a.barh(names, [r2[m] for m in order], color=colors, height=0.66, edgecolor="white", linewidth=2)
a.axvspan(0, r2[normal], color="#f0efec", zorder=0)
a.axvline(r2[normal], color=MUTED, ls="--", lw=1.6, zorder=1)
a.text(r2[normal] / 2, top + 0.05, f"free from the normal\nalone ({r2[normal]:.2f})", ha="center", va="center", color=MUTED, fontsize=12.5)
for i, m in enumerate(order):
    a.text(r2[m] + 0.012, i, f"{r2[m]:.2f}", va="center", fontsize=13.5, color=INK)
a.set_xlim(0, 0.8)
a.set_ylim(-0.6, N + 0.55)
a.set_xlabel("R² (share of the temperature variation explained)")
a.grid(axis="y", visible=False)
a.get_yticklabels()[order.index("MOS (ECMWF + station)")].set_fontweight("bold")
a.set_title("How much each model explains", loc="left", fontsize=15.5, pad=10)

b.barh(names, [skill[m] for m in order], color=colors, height=0.66, edgecolor="white", linewidth=2)
b.axvline(0, color=INK, lw=1.4)
for i, m in enumerate(order):
    v = skill[m]
    b.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=13.5, color=INK)
b.set_xlim(-0.55, 0.55)
b.text(-0.27, top + 0.05, "worse than just the normal", ha="center", va="center", color=MUTED, fontsize=12.5)
b.text(0.27, top + 0.05, "better than the normal", ha="center", va="center", color=MUTED, fontsize=12.5)
b.set_xlabel("Skill over the normal (share of its error removed)")
b.grid(axis="y", visible=False)
b.set_title("What each model adds beyond the normal", loc="left", fontsize=15.5, pad=10)

seen = {}
for m in order[::-1]:
    seen.setdefault(*style_of(m))
fig.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()], loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=2, fontsize=12.5,
           frameon=False)
fig.text(0.01, 0.955, "R² looks high for everyone because the normal already explains 43%", fontsize=19, fontweight="bold", ha="left", va="center")
fig.text(0.01, 0.905, "The real question is what a model adds on top of it: only MOS, raw ECMWF and the random forest clearly improve on the normal",
         fontsize=13.5, color=MUTED, ha="left", va="center")
footnote(fig, "R² = 1 − (model's squared error) / (squared error of predicting the window's average). Skill over the normal = 1 − (model's squared error) / "
              "(squared error of the normal for the date). 322 real hours, Sep 17–30 2026.")
fig.savefig(OUT / "fig9_r2_skill.png", dpi=200, facecolor="white")
plt.close(fig)
print("saved", OUT / "fig9_r2_skill.png")
print({label.get(m, m): (round(r2[m], 3), round(skill[m], 3)) for m in order[::-1]})
