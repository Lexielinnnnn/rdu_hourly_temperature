"""Time-ordered cross-validation ("show your work") for the two final models.

Folds: one per year (2022-2025).  Each fold trains ONLY on forecasts whose 14-day window ends before Aug 15 of that
year, then is scored on 21 forecasts (every 3 days, Aug 15 - Oct 15, issued at local midnight).  Training error is
measured on the same Aug 15 - Oct 15 season of the training years, so it is comparable to validation error.
The real Sep 17-30 2026 period is never touched here.

What it produces (final_model/results/):
  cv_rf_depth.csv       random forest: training vs validation error as tree depth grows (validation curve)
  cv_linear_alpha.csv   Linear model: training vs validation error as the ridge penalty grows
  cv_linear_groups.csv  Feature-group check (temp / + dew point / + pressure)
  cv_alignment.csv      the random forest fix: training forecasts at local midnight vs UTC midnight (original push)
  cv_model_comparison*.csv, cv_error_by_lead_day.csv, chosen.json

Run from repo root:  python -W ignore final_model/cross_validation.py     (about 10-20 minutes)
Speed note: the random forest curves use 60 trees / 30% bootstrap / every 2nd training day; the final model
(train_final.py) uses the notebook's full settings at the depth chosen here.
"""
import json
import time

import numpy as np
import pandas as pd

import linear_model as LM
import rdu_data as R
import rf_model as RF

YEARS = [2022, 2023, 2024, 2025]
RF_DEPTHS = [2, 4, 6, 8, 10, 12, 16, 20]
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]
GROUPSETS = {"temp": ("temp",), "temp + dew point": ("temp", "dewpoint"), "temp + dew point + pressure": ("temp", "dewpoint", "pressure")}
RF_CV = dict(n_estimators=60, max_samples=0.3, min_samples_leaf=20, max_features=0.8)
RF_EVERY, TRAIN_SAMPLE, NOTEBOOK_DEPTH = 2, 6000, 16

data = R.load()
hourly = R.hourly_grid(data)
temp_utc = hourly["temperature"]
temp_local = pd.Series(temp_utc.values, index=temp_utc.index.tz_convert(R.TZ))
cache = {}


def fold_origins(yr):
    local = pd.date_range(f"{yr}-08-15", f"{yr}-10-15", freq="3D", tz=R.TZ)
    return local, local.tz_convert("UTC")


def actual_and_lead(utc_origins):
    act = np.concatenate([temp_utc.reindex(pd.date_range(o, periods=R.HORIZON, freq="h")).to_numpy() for o in utc_origins])
    return act, np.tile(np.arange(R.HORIZON), len(utc_origins))


def feats(o):
    if o not in cache:
        cache[o] = RF.make_features(data, o)
    return cache[o]


rows, preds, truth = [], {}, {}
for yr in YEARS:
    truth[yr] = actual_and_lead(fold_origins(yr)[1])


def record(model, hyper_name, hyper, yr, p, train_mae):
    act, _ = truth[yr]
    rows.append(dict(model=model, hyper_name=hyper_name, hyper=hyper, fold=yr, train_mae=train_mae,
                     val_mae=R.mae(act, p), val_rmse=R.rmse(act, p)))
    preds[(model, str(hyper), yr)] = p
    print(f"{model:24s} {hyper_name}={hyper!s:28s} fold {yr}: train {train_mae:.2f}  val {rows[-1]['val_mae']:.2f}", flush=True)


def run_rf(label, align, depth, yr):
    local, utc = fold_origins(yr)
    X, y = RF.make_training_data(data, utc[0], align=align, every=RF_EVERY, cache=cache)
    m = RF.make_model(max_depth=depth, **RF_CV).fit(X, y)
    loc = X.index.tz_convert(R.TZ)
    md = loc.month * 100 + loc.day
    pool = np.flatnonzero((md >= 815) & (md <= 1015))
    idx = np.random.default_rng(0).choice(pool, size=min(TRAIN_SAMPLE, len(pool)), replace=False)
    train_mae = R.mae(y.iloc[idx], m.predict(X.iloc[idx]))
    Xv = pd.concat([feats(o) for o in utc])
    record(label, "max_depth", depth, yr, m.predict(Xv), train_mae)


t0 = time.time()
# ---------------- two-stage linear model ----------------
for yr in YEARS:
    local, utc = fold_origins(yr)
    for gname, groups in GROUPSETS.items():
        m = LM.LinearForecaster(hourly, groups, alpha=1.0).fit(utc[0])
        record("Linear (two-stage)", "groups", gname, yr, m.predict_many(local).ravel(), m.train_error_sample())
        if gname == "temp":
            preds[("Fourier climatology only", "-", yr)] = np.concatenate([m.climatology_only(o) for o in local])
            preds[("Climatology baseline (same hour, +/-7 days)", "-", yr)] = np.concatenate([LM.climatology_baseline(temp_local, o) for o in local])
            preds[("Persistence (repeat last 24 h)", "-", yr)] = np.concatenate([LM.persistence_baseline(temp_local, o) for o in local])
df = pd.DataFrame(rows)
g = df[df.hyper_name == "groups"].groupby("hyper")["val_mae"].mean()
best_groups = next(k for k in GROUPSETS if g[k] <= g.min() + 0.01)          # simplest set within 0.01 C of the best
for yr in YEARS:
    local, utc = fold_origins(yr)
    m = LM.LinearForecaster(hourly, GROUPSETS[best_groups]).fit(utc[0])
    for a in ALPHAS:
        m.set_alpha(a)
        record("Linear (two-stage)", "alpha", a, yr, m.predict_many(local).ravel(), m.train_error_sample())
df = pd.DataFrame(rows)
best_alpha = next(a for a in ALPHAS if a == float(df[df.hyper_name == "alpha"].groupby("hyper")["val_mae"].mean().astype(float).idxmin()))
print(f"linear done in {(time.time() - t0) / 60:.1f} min: groups={best_groups}, alpha={best_alpha}", flush=True)

# ---------------- random forest (fixed: local-midnight training) ----------------
for depth in RF_DEPTHS:
    for yr in YEARS:
        run_rf("RF (fixed)", "local", depth, yr)
df = pd.DataFrame(rows)
rf = df[(df.model == "RF (fixed)")].groupby("hyper")["val_mae"].mean()
best_depth = int(rf.idxmin())
# the pushed behavior (UTC-midnight training) at the notebook depth and at the chosen depth, to measure the fix
for depth in sorted({NOTEBOOK_DEPTH, best_depth}):
    for yr in YEARS:
        run_rf("RF (original push)", "utc", depth, yr)
print(f"rf done in {(time.time() - t0) / 60:.1f} min: depth={best_depth}", flush=True)

# ---------------- save tables ----------------
df = pd.DataFrame(rows)
df.to_csv("final_model/results/cv_all_runs.csv", index=False)


def curve(model, hp, name):
    d = df[(df.model == model) & (df.hyper_name == hp)].copy()
    if hp != "groups":
        d["hyper"] = d["hyper"].astype(float)
    g = d.groupby("hyper", sort=False)[["train_mae", "val_mae", "val_rmse"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g.to_csv(f"final_model/results/{name}.csv")


curve("RF (fixed)", "max_depth", "cv_rf_depth")
curve("Linear (two-stage)", "alpha", "cv_linear_alpha")
curve("Linear (two-stage)", "groups", "cv_linear_groups")
al = df[df.model.isin(["RF (fixed)", "RF (original push)"]) & df.hyper.isin([NOTEBOOK_DEPTH, best_depth])].copy()
al["training forecasts start at"] = np.where(al.model == "RF (fixed)", "local midnight (fixed)", "UTC midnight (original push)")
ag = al.groupby(["hyper", "training forecasts start at"])[["val_mae", "val_rmse"]].agg(["mean", "std"])
ag.columns = ["_".join(c) for c in ag.columns]
ag.reset_index().rename(columns={"hyper": "max_depth"}).to_csv("final_model/results/cv_alignment.csv", index=False)

# model comparison at the chosen settings, same folds and hours for everyone
series = {
    "Linear (two-stage)": lambda yr: preds[("Linear (two-stage)", str(best_alpha), yr)],
    "Random forest (fixed, chosen depth)": lambda yr: preds[("RF (fixed)", str(best_depth), yr)],
    "Random forest (fixed, notebook depth 16)": lambda yr: preds[("RF (fixed)", str(NOTEBOOK_DEPTH), yr)],
    "Random forest (original push, UTC midnight, depth 16)": lambda yr: preds[("RF (original push)", str(NOTEBOOK_DEPTH), yr)],
    "Fourier climatology only": lambda yr: preds[("Fourier climatology only", "-", yr)],
    "Climatology baseline (same hour, +/-7 days)": lambda yr: preds[("Climatology baseline (same hour, +/-7 days)", "-", yr)],
    "Persistence (repeat last 24 h)": lambda yr: preds[("Persistence (repeat last 24 h)", "-", yr)],
}
comp, lead_rows = [], []
for name, get in series.items():
    for yr in YEARS:
        act, lead = truth[yr]
        p = get(yr)
        comp.append(dict(model=name, fold=yr, mae=R.mae(act, p), rmse=R.rmse(act, p)))
        for d_ in range(14):
            s = (lead >= 24 * d_) & (lead < 24 * (d_ + 1)) & ~np.isnan(act)
            lead_rows.append(dict(model=name, lead_day=d_ + 1, abs_err=np.abs(act[s] - p[s]).sum(), n=int(s.sum())))
comp = pd.DataFrame(comp)
comp.to_csv("final_model/results/cv_model_comparison_by_fold.csv", index=False)
summ = comp.groupby("model")[["mae", "rmse"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.sort_values("mae_mean").to_csv("final_model/results/cv_model_comparison.csv")
ld = pd.DataFrame(lead_rows).groupby(["model", "lead_day"])[["abs_err", "n"]].sum()
(ld.abs_err / ld.n).rename("mae").reset_index().to_csv("final_model/results/cv_error_by_lead_day.csv", index=False)
json.dump({"rf": {"max_depth": best_depth, "notebook_max_depth": NOTEBOOK_DEPTH},
           "linear": {"groups": list(GROUPSETS[best_groups]), "groups_name": best_groups, "alpha": best_alpha}},
          open("final_model/results/chosen.json", "w"), indent=2)
print(summ.sort_values("mae_mean").round(3).to_string())
print(open("final_model/results/chosen.json").read())
