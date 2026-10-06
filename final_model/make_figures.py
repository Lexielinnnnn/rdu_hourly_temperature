"""Slide-ready cross-validation figures from final_model/results/cv_*.csv.
Run from repo root after cross_validation.py:  python -W ignore final_model/make_figures.py
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import AQUA, BLUE, GREY, INK, MUTED, ORANGE, bar_title, footnote, legend_below, save

R = "final_model/results/"
chosen = json.load(open(R + "chosen.json"))
FOOT = "Band = ±1 std across 4 yearly folds; each fold trains only on data before its Aug 15 start."


def curve(csv, xlabel, title, best, name, log=False, fmt=lambda v: f"{v:g}", extra=""):
    d = pd.read_csv(csv)
    d["hyper"] = pd.to_numeric(d["hyper"])
    x = d["hyper"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for col, color, label in [("train_mae", BLUE, "Training error"), ("val_mae", ORANGE, "Validation error (held-out seasons)")]:
        m, s = d[col + "_mean"].to_numpy(), d[col + "_std"].fillna(0).to_numpy()
        ax.plot(x, m, "-o", color=color, lw=2.5, ms=7, label=label, zorder=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    if log:
        ax.set_xscale("log")
    ax.axvline(best, color=MUTED, ls="--", lw=1.5, zorder=2)
    ax.text(best, ax.get_ylim()[1], f"  chosen: {fmt(best)}", color=INK, va="top", ha="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean absolute error (°C)"); ax.set_title(title, loc="left")
    legend_below(ax, 2)
    footnote(fig, FOOT + extra)
    save(fig, name)


curve(R + "cv_rf_depth.csv", "Maximum tree depth   [more complex model on the right]",
      "Lexie's random forest: training vs validation error", chosen["rf"]["max_depth"], "fig_cv_rf_depth.png",
      fmt=lambda v: f"depth {v:g}", extra=" Notebook depth was 16.")
curve(R + "cv_linear_alpha.csv", "Ridge penalty (alpha), log scale   [more complex model on the left]",
      "Burak's linear model: training vs validation error", chosen["linear"]["alpha"], "fig_cv_linear_alpha.png", log=True,
      extra=" Flat curve = the penalty barely matters.")

# ---- Burak's feature-group check ----
d = pd.read_csv(R + "cv_linear_groups.csv").sort_values("val_mae_mean", ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh(d["hyper"], d["val_mae_mean"], xerr=d["val_mae_std"].fillna(0), height=0.55,
        color=[BLUE if h == chosen["linear"]["groups_name"] else GREY for h in d["hyper"]],
        error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
for i, (v, sd) in enumerate(zip(d["val_mae_mean"], d["val_mae_std"].fillna(0))):
    ax.text(v + sd + 0.04, i, f"{v:.2f}", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(d["val_mae_mean"].min() - 0.5, d["val_mae_mean"].max() + 0.6)
ax.set_xlabel("Validation MAE (°C), lower is better"); bar_title(fig, "Linear model: do dew point and pressure help?")
ax.grid(axis="y", visible=False)
save(fig, "fig_cv_linear_groups.png", top=0.9)

# ---- comparison of all models and baselines ----
c = pd.read_csv(R + "cv_model_comparison.csv").sort_values("mae_mean", ascending=False)


def color(n):
    return BLUE if n.startswith("Linear") else ORANGE if n.startswith("Random forest") else GREY


fig, ax = plt.subplots(figsize=(11.5, 5.6))
bars = ax.barh(c["model"], c["mae_mean"], xerr=c["mae_std"], height=0.6, color=[color(n) for n in c["model"]],
               error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
for b, n in zip(bars, c["model"]):
    if "original push" in n:
        b.set_hatch("//"); b.set_edgecolor("white")
for i, (v, sd) in enumerate(zip(c["mae_mean"], c["mae_std"])):
    ax.text(v + sd + 0.05, i, f"{v:.2f}", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(0, c["mae_mean"].max() + c["mae_std"].max() + 0.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("Validation MAE (°C): mean of 4 yearly folds ±1 std"); bar_title(fig, "Cross-validation: final candidates vs baselines")
footnote(fig, "Hatched = random forest trained the way the notebook was first pushed (UTC-midnight forecasts, before Lexie's fix).")
save(fig, "fig_cv_model_comparison.png", top=0.93)

# ---- the random forest alignment fix ----
a = c[c["model"].str.startswith("Random forest")].sort_values("model")
fig, ax = plt.subplots(figsize=(11, 3.8))
bars = ax.barh(a["model"], a["mae_mean"], xerr=a["mae_std"], height=0.55, color=ORANGE, error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
for b, n in zip(bars, a["model"]):
    if "original push" in n:
        b.set_hatch("//"); b.set_edgecolor("white"); b.set_facecolor(GREY)
for i, (v, sd) in enumerate(zip(a["mae_mean"], a["mae_std"])):
    ax.text(v + sd + 0.03, i, f"{v:.3f}", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(a["mae_mean"].min() - 0.5, a["mae_mean"].max() + 0.6)
ax.set_xlabel("Validation MAE (°C)"); bar_title(fig, "Random forest: training forecasts at local midnight vs UTC midnight", 15)
ax.grid(axis="y", visible=False)
save(fig, "fig_cv_rf_alignment.png", top=0.88)

# ---- error by lead day ----
l = pd.read_csv(R + "cv_error_by_lead_day.csv")
fig, ax = plt.subplots(figsize=(10, 5.4))
for model, col, ls in [("Linear (Burak's two-stage)", BLUE, "-"), ("Random forest (fixed, chosen depth)", ORANGE, "-"),
                       ("Climatology baseline (same hour, +/-7 days)", GREY, "--")]:
    s = l[l.model == model]
    ax.plot(s.lead_day, s.mae, ls, color=col, lw=2.5, marker="o", ms=5, label=model)
ax.set_xticks(range(1, 15)); ax.set_xlabel("Forecast day"); ax.set_ylabel("MAE (°C)")
ax.set_title("Error by forecast day (cross-validation)", loc="left")
legend_below(ax, 1, offset=-0.17)
save(fig, "fig_cv_error_by_lead_day.png")
