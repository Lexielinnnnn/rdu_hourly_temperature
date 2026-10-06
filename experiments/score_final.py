"""FINAL SCORING: compare predictions with the real Sep 17-30 2026 RDU observations and save slide figures.

Run this ONCE, after model_selection.py and final_forecast.py.  The real observations are used here and
nowhere else; do not tune anything afterwards (that would turn the test set into a validation set).
The pushed models are scored as they are (rdu_linear_predictions.csv, rdu_random_forest_predictions.csv).
Run from repo root:  python -W ignore experiments/score_final.py
"""
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline as P
from plot_style import AQUA, BLUE, GREY, INK, MUTED, ORANGE, footnote, legend_below, save

chosen = json.load(open("results/chosen_models.json"))
other = chosen["other_model"]
OTHER_LABEL = {"knn": "KNN", "rf": "Random forest"}[other]

# ---- real observations (same nearest-:51 hourly rule as the training data) ----
# raw_data/final_truth/rdu_2026.psv comes from experiments/fetch_final_truth.sh (NOAA GHCNh); read ONLY here
raw = pd.read_csv("raw_data/final_truth/rdu_2026.psv", sep="|", usecols=["DATE", "temperature"], low_memory=False)
raw["DATE"] = pd.to_datetime(raw["DATE"], utc=True)
raw["temperature"] = pd.to_numeric(raw["temperature"], errors="coerce")
hours = pd.date_range(P.CUTOFF, periods=P.H, freq="h")
actual = P.hourly(raw.set_index("DATE", drop=False))["temperature"].reindex(hours)
print(f"actual hours available: {actual.notna().sum()} of {len(hours)}; last scored hour: {actual.dropna().index.max().tz_convert(P.TZ)}")
assert actual.notna().sum() > 200, "real observations missing - run experiments/fetch_final_truth.sh"


def load_pred(text_or_path, is_text=False):
    df = pd.read_csv(StringIO(text_or_path) if is_text else text_or_path)
    df["h"] = pd.to_datetime(df["hour_local"], utc=True)
    return df.set_index("h")["predicted_temperature"].reindex(hours)


preds = {"Ridge linear regression (ours)": load_pred("predictions_ridge.csv"),
         "KNN (ours)": load_pred("predictions_knn.csv"),
         "Random forest (ours)": load_pred("predictions_rf.csv")}
extra = {}
if Path("rdu_linear_predictions.csv").exists():
    extra["Linear model (as pushed)"] = load_pred("rdu_linear_predictions.csv")
if Path("rdu_random_forest_predictions.csv").exists():
    extra["Random forest (as pushed)"] = load_pred("rdu_random_forest_predictions.csv")

S = P.load_sources()
O, Y, G = P.build_features(S)
X_fc, _ = P.make_features(O, Y, S["rdu"]["temperature"], P.CUTOFF)
last24 = S["rdu"]["temperature"].reindex(pd.date_range(P.CUTOFF - pd.Timedelta(hours=24), periods=24, freq="h")).interpolate(limit_direction="both")
base = {"Baseline: same hour, last 2 yrs": pd.Series(X_fc["yp_mean_sm7"].to_numpy(), index=hours),
        "Baseline: repeat last 24 h": pd.Series(np.tile(last24.to_numpy(), 14), index=hours)}

allp = {**preds, **extra, **base}
ok = actual.notna()
rows, day_rows = [], []
sst = ((actual[ok] - actual[ok].mean()) ** 2).sum()
for name, p in allp.items():
    e = (p[ok] - actual[ok])
    rows.append(dict(model=name, MAE_C=e.abs().mean(), RMSE_C=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / sst,
                     bias_C=e.mean(), MAE_F=1.8 * e.abs().mean(), RMSE_F=1.8 * np.sqrt((e ** 2).mean()), n_hours=int(ok.sum())))
    for d in range(14):
        sel = ok.to_numpy() & (np.arange(len(hours)) // 24 == d)
        day_rows.append(dict(model=name, day=d + 1, MAE_C=float(np.abs(p[sel] - actual[sel]).mean())))
m = pd.DataFrame(rows).sort_values("MAE_C")
m.to_csv("results/final_metrics.csv", index=False)
pd.DataFrame(day_rows).to_csv("results/final_metrics_by_day.csv", index=False)
pd.DataFrame({"actual": actual, **allp}).tz_convert(P.TZ).to_csv("results/final_forecast_vs_actual.csv")
print(m.round(3).to_string(index=False))

# ---- figures ----
colors = {n: (BLUE if n.startswith("Ridge") else ORANGE if n.startswith("KNN") else AQUA if n.startswith("Random") else GREY)
          for n in allp}
mm = m.sort_values("MAE_C", ascending=False)
fig, ax = plt.subplots(figsize=(11, 5.8))
bars = ax.barh(mm["model"], mm["MAE_C"], color=[colors[n] for n in mm["model"]], height=0.6)
for b, n in zip(bars, mm["model"]):
    if "as pushed" in n:
        b.set_hatch("//"); b.set_edgecolor("white")
for i, (v, f_) in enumerate(zip(mm["MAE_C"], mm["MAE_F"])):
    ax.text(v + 0.03, i, f"{v:.2f} °C  ({f_:.2f} °F)", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(0, mm["MAE_C"].max() + 1.2); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean absolute error on the real Sep 17-30 2026 hours"); ax.set_title("Final test: every model vs the real Sep 17-30", loc="left")
footnote(fig, "Hatched = teammate models, as pushed to GitHub. Real observations used only for this final scoring.")
save(fig, "fig_final_model_comparison.png")

fig, ax = plt.subplots(figsize=(13, 5.4))
ax.plot(hours.tz_convert(P.TZ), actual, color=INK, lw=2.2, label="Actual")
ax.plot(hours.tz_convert(P.TZ), preds["Ridge linear regression (ours)"], color=BLUE, lw=2, label="Ridge linear regression")
okey = f"{OTHER_LABEL} (ours)"
ax.plot(hours.tz_convert(P.TZ), preds[okey], color=ORANGE, lw=2, label=OTHER_LABEL)
ax.set_ylabel("Temperature (°C)"); ax.set_title("RDU hourly temperature, Sep 17-30 2026: forecast vs actual", loc="left")
fig.autofmt_xdate()
legend_below(ax, 3, offset=-0.32)
save(fig, "fig_final_forecast_vs_actual.png")

dd = pd.DataFrame(day_rows)
fig, ax = plt.subplots(figsize=(10, 5.4))
for name, color, ls in [("Ridge linear regression (ours)", BLUE, "-"), (okey, ORANGE, "-"), ("Baseline: same hour, last 2 yrs", GREY, "--")]:
    s = dd[dd.model == name]
    ax.plot(s.day, s.MAE_C, ls, color=color, lw=2.5, marker="o", ms=5, label=name.replace(" (ours)", ""))
ax.set_xticks(range(1, 15)); ax.set_xlabel("Forecast day (Sep 17 = day 1)"); ax.set_ylabel("MAE (°C)")
ax.set_title("Final test: error by forecast day", loc="left")
legend_below(ax, 3)
save(fig, "fig_final_error_by_day.png")

# ---- what the models learned ----
co = pd.read_csv("results/ridge_coefficients.csv", index_col=0).iloc[:, 0]
top = co.reindex(co.abs().sort_values(ascending=False).index).head(12)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top.index, top.values, color=[BLUE if v > 0 else ORANGE for v in top.values], height=0.65)
ax.set_xlabel("Standardized coefficient (°C per 1 std of the feature)")
ax.set_title("Ridge: 12 largest coefficients (blue raises, orange lowers the forecast)", loc="left", fontsize=15)
ax.grid(axis="y", visible=False)
save(fig, "fig_ridge_coefficients.png")
imp = pd.read_csv("results/rf_importances.csv", index_col=0).iloc[:, 0].head(12)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp.index, imp.values, color=AQUA, height=0.65)
ax.set_xlabel("Feature importance (share of variance reduction)"); ax.set_title("Random forest: 12 most important features", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_rf_importances.png")
