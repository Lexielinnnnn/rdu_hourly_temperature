"""Slide-ready figures from results/cv_*.csv (cross-validation on past Aug 15 - Oct 15 seasons).
Run from repo root after model_selection.py:  python -W ignore experiments/make_cv_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import AQUA, BLUE, GREY, INK, MUTED, ORANGE, OUT, footnote, legend_below, save

chosen = json.load(open("results/chosen_models.json"))


def curve(csv, xlabel, title, best, name, log=False, note=None, fmt=lambda v: f"{v:g}"):
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
    ymax = ax.get_ylim()[1]
    ax.text(best, ymax, f"  chosen: {fmt(best)}", color=INK, va="top", ha="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean absolute error (°C)"); ax.set_title(title, loc="left")
    legend_below(ax, 2)
    footnote(fig, "Band = ±1 std across 4 yearly folds; each fold trains only on data before its Aug 15 start." + (f" {note}" if note else ""))
    save(fig, name)


curve("results/cv_ridge_alpha.csv", "Ridge penalty (alpha), log scale   [more complex model on the left]",
      "Ridge linear regression: training vs validation error", chosen["ridge"]["alpha"], "fig_cv_ridge_alpha.png", log=True)
curve("results/cv_knn_k.csv", "Number of neighbors (k), log scale   [more complex model on the left]",
      "KNN: training vs validation error", chosen["knn"]["k"], "fig_cv_knn_k.png", log=True,
      note="Training error at small k is optimistic: each point counts itself as a neighbor.")
curve("results/cv_rf_max_depth.csv", "Maximum tree depth   [more complex model on the right]",
      "Random forest: training vs validation error", chosen["rf"]["max_depth"], "fig_cv_rf_depth.png", fmt=lambda v: f"depth {v:g}")

# ---- ridge feature groups ----
d = pd.read_csv("results/cv_ridge_features.csv").sort_values("val_mae_mean", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = [ORANGE if h == chosen["ridge"]["features"] else GREY for h in d["hyper"]]
ax.barh(d["hyper"], d["val_mae_mean"], xerr=d["val_mae_std"].fillna(0), color=colors, height=0.6,
        error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
for i, v in enumerate(d["val_mae_mean"]):
    ax.text(v + 0.03, i, f"{v:.2f}", va="center", color=INK, fontsize=12)
ax.set_xlim(left=max(0, d["val_mae_mean"].min() - 0.6))
ax.set_xlabel("Validation MAE (°C), lower is better"); ax.set_title("Which data helps the linear model?", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_cv_ridge_feature_groups.png")

# ---- model comparison ----
c = pd.read_csv("results/cv_model_comparison.csv").sort_values("mae_mean", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5.2))
col = {"Ridge linear regression": BLUE, "KNN": ORANGE, "Random forest": AQUA}
ax.barh(c["model"], c["mae_mean"], xerr=c["mae_std"], height=0.6, color=[col.get(m, GREY) for m in c["model"]],
        error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
for i, v in enumerate(c["mae_mean"]):
    ax.text(v + 0.04, i, f"{v:.2f}", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(0, c["mae_mean"].max() + c["mae_std"].max() + 0.7)
ax.set_xlabel("Validation MAE (°C), mean of 4 yearly folds ±1 std"); ax.set_title("Tuned models vs baselines", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_cv_model_comparison.png")

# ---- error by lead day ----
l = pd.read_csv("results/cv_error_by_lead_day.csv")
fig, ax = plt.subplots(figsize=(10, 5.4))
for model, color, ls in [("Ridge linear regression", BLUE, "-"), ("KNN", ORANGE, "-"), ("Random forest", AQUA, "-"),
                         ("Baseline: same hour, last 2 yrs", GREY, "--")]:
    s = l[l.model == model]
    ax.plot(s.lead_day, s.mae, ls, color=color, lw=2.5, marker="o", ms=5, label=model)
ax.set_xticks(range(1, 15)); ax.set_xlabel("Forecast day"); ax.set_ylabel("MAE (°C)")
ax.set_title("Error grows with forecast distance, then flattens", loc="left")
legend_below(ax, 2)
save(fig, "fig_cv_error_by_lead_day.png")
