"""FINAL SCORING of the two final models against the real Sep 17-30 2026 RDU observations.

Run this ONCE, after cross_validation.py and train_final.py, and do not tune anything afterwards.
Real observations: raw_data/final_truth/rdu_2026.psv (bash experiments/fetch_final_truth.sh).  Only this script reads them.
Run from repo root:  python -W ignore final_model/score_final.py
"""
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import linear_model as LM
import rdu_data as R
from plot_style import AQUA, BLUE, GREY, INK, MUTED, ORANGE, bar_title, footnote, legend_below, save

chosen = json.load(open("final_model/results/chosen.json"))
hours = pd.date_range(R.CUTOFF, periods=R.HORIZON, freq="h")
hours_local = hours.tz_convert(R.TZ)

# ---- real observations, same "report nearest :51" rule as the training data ----
raw = pd.read_csv("raw_data/final_truth/rdu_2026.psv", sep="|", usecols=["DATE", "temperature"], low_memory=False)
raw["DATE"] = pd.to_datetime(raw["DATE"], utc=True)
raw["temperature"] = pd.to_numeric(raw["temperature"], errors="coerce")
raw = raw.dropna(subset=["temperature"])
raw["hour"] = raw["DATE"].dt.floor("h")
raw["dist"] = (raw["DATE"].dt.minute - 51).abs()
actual = raw.sort_values(["hour", "dist"]).drop_duplicates("hour").set_index("hour")["temperature"].reindex(hours)
ok = actual.notna().to_numpy()
assert ok.sum() > 200, "real observations missing - run experiments/fetch_final_truth.sh"
print(f"real hours available: {ok.sum()} of {len(hours)}; last scored hour: {actual.dropna().index.max().tz_convert(R.TZ)}")


def load_pred(src, text=False):
    df = pd.read_csv(StringIO(src) if text else src)
    return pd.Series(df["predicted_temperature"].to_numpy(), index=pd.to_datetime(df["hour_local"], utc=True)).reindex(hours).to_numpy()


P = {"Random forest (Lexie's, fixed)": load_pred("final_model/predictions/rf_predictions.csv"),
     "Linear (Burak's two-stage)": load_pred("final_model/predictions/linear_predictions.csv")}
P["Average of the two"] = (P["Random forest (Lexie's, fixed)"] + P["Linear (Burak's two-stage)"]) / 2

# reference rows: the models exactly as pushed to GitHub, and baselines
ref = {}
if Path("rdu_random_forest_predictions.csv").exists():      # Lexie's latest push (d29c7ed: local-midnight fix, depth 16)
    ref["Random forest (Lexie, latest push)"] = load_pred("rdu_random_forest_predictions.csv")
try:                                                       # her first push (854e0d9: UTC-midnight training)
    ref["Random forest (Lexie, original push)"] = load_pred(subprocess.run(
        ["git", "show", "854e0d9:rdu_random_forest_predictions.csv"], capture_output=True, text=True, check=True).stdout, True)
except Exception as e:
    print("original random forest predictions not available:", e)
try:
    ref["Linear (Burak, as pushed)"] = load_pred(subprocess.run(["git", "show", "origin/burak:rdu_linear_predictions.csv"],
                                                                capture_output=True, text=True, check=True).stdout, True)
except Exception as e:
    print("burak's pushed predictions not available:", e)
data = R.load()
hourly = R.hourly_grid(data)
temp_local = pd.Series(hourly["temperature"].to_numpy(), index=hourly.index.tz_convert(R.TZ))
origin_local = R.CUTOFF.tz_convert(R.TZ)
lin = LM.LinearForecaster(hourly, chosen["linear"]["groups"], alpha=chosen["linear"]["alpha"]).fit(R.CUTOFF)
base = {"Baseline: Fourier climatology only": lin.climatology_only(R.CUTOFF),
        "Baseline: same hour, +/-7 days of prior years": LM.climatology_baseline(temp_local, origin_local),
        "Baseline: repeat last 24 h": LM.persistence_baseline(temp_local, origin_local)}
allp = {**P, **ref, **base}

a = actual.to_numpy()
sst = np.sum((a[ok] - a[ok].mean()) ** 2)
rows, day_rows = [], []
for name, p in allp.items():
    e = p[ok] - a[ok]
    rows.append(dict(model=name, MAE_C=np.abs(e).mean(), RMSE_C=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / sst,
                     bias_C=e.mean(), MAE_F=1.8 * np.abs(e).mean(), RMSE_F=1.8 * np.sqrt((e ** 2).mean()), n_hours=int(ok.sum())))
    for d in range(14):
        s = ok & (np.arange(len(hours)) // 24 == d)
        day_rows.append(dict(model=name, day=d + 1, MAE_C=float(np.abs(p[s] - a[s]).mean())))
m = pd.DataFrame(rows).sort_values("MAE_C")
m.to_csv("final_model/results/final_metrics.csv", index=False)
pd.DataFrame(day_rows).to_csv("final_model/results/final_metrics_by_day.csv", index=False)
pd.DataFrame({"actual": a, **allp}, index=hours_local).to_csv("final_model/results/final_forecast_vs_actual.csv")
print(m.round(3).to_string(index=False))

# ---- figures ----
def col(n):
    return BLUE if n.startswith("Linear") else ORANGE if n.startswith("Random") else AQUA if n.startswith("Average") else GREY


mm = m.sort_values("MAE_C", ascending=False)
fig, ax = plt.subplots(figsize=(11.5, 6))
bars = ax.barh(mm["model"], mm["MAE_C"], color=[col(n) for n in mm["model"]], height=0.6)
for b, n in zip(bars, mm["model"]):
    if "push" in n:
        b.set_hatch("//"); b.set_edgecolor("white")
for i, (v, f_) in enumerate(zip(mm["MAE_C"], mm["MAE_F"])):
    ax.text(v + 0.03, i, f"{v:.2f} °C  ({f_:.2f} °F)", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(0, mm["MAE_C"].max() + 1.3); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean absolute error on the real Sep 17-30 2026 hours"); bar_title(fig, "Final test: our two models vs baselines")
footnote(fig, "Hatched = teammates' models exactly as pushed to GitHub. Real observations used only for final scoring.")
save(fig, "fig_final_model_comparison.png", top=0.93)

fig, ax = plt.subplots(figsize=(13, 5.4))
ax.plot(hours_local, a, color=INK, lw=2.2, label="Actual")
ax.plot(hours_local, P["Linear (Burak's two-stage)"], color=BLUE, lw=2, label="Linear (Burak)")
ax.plot(hours_local, P["Random forest (Lexie's, fixed)"], color=ORANGE, lw=2, label="Random forest (Lexie)")
ax.set_ylabel("Temperature (°C)"); ax.set_title("RDU hourly temperature, Sep 17-30 2026: forecast vs actual", loc="left")
fig.autofmt_xdate()
legend_below(ax, 3, offset=-0.32)
save(fig, "fig_final_forecast_vs_actual.png")

dd = pd.DataFrame(day_rows)
fig, ax = plt.subplots(figsize=(10, 5.4))
for name, c, ls in [("Linear (Burak's two-stage)", BLUE, "-"), ("Random forest (Lexie's, fixed)", ORANGE, "-"),
                    ("Baseline: same hour, +/-7 days of prior years", GREY, "--")]:
    s = dd[dd.model == name]
    ax.plot(s.day, s.MAE_C, ls, color=c, lw=2.5, marker="o", ms=5, label=name)
ax.set_xticks(range(1, 15)); ax.set_xlabel("Forecast day (Sep 17 = day 1)"); ax.set_ylabel("MAE (°C)")
ax.set_title("Final test: error by forecast day", loc="left")
legend_below(ax, 1, offset=-0.17)
save(fig, "fig_final_error_by_day.png")

co = pd.read_csv("final_model/results/linear_coefficients.csv", index_col=0).iloc[:, 0]
top = co.reindex(co.abs().sort_values(ascending=False).index).head(12)[::-1]
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top.index, top.values, color=[BLUE if v > 0 else ORANGE for v in top.values], height=0.65)
ax.set_xlabel("Standardized coefficient (°C per 1 std of the feature)")
ax.set_title("Linear model: 12 largest coefficients (blue raises, orange lowers the forecast)", loc="left", fontsize=15)
ax.grid(axis="y", visible=False)
save(fig, "fig_linear_coefficients.png")
imp = pd.read_csv("final_model/results/rf_importances.csv", index_col=0).iloc[:, 0].head(12)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp.index, imp.values, color=ORANGE, height=0.65)
ax.set_xlabel("Feature importance (share of variance reduction)"); ax.set_title("Random forest: 12 most important features", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_rf_importances.png")
