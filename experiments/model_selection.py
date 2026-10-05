"""Time-ordered cross-validation and validation curves for ridge, KNN and random forest.

Folds: one per year (2022-2025).  For each fold the model is trained ONLY on forecast origins whose 14-day
target window ends before the fold's first validation origin (Aug 15), then scored on 21 forecasts
(every 3 days, Aug 15 - Oct 15).  Training error is measured on a random sample (up to 6k rows) of the same
Aug 15 - Oct 15 season in the training years.  The real Sep 17-30 2026 forecast period is never touched here.

Outputs: results/cv_*.csv and results/chosen_models.json (consumed by final_forecast.py and make_cv_figures.py).
Run from repo root:  python -W ignore experiments/model_selection.py
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import modeling as M
import pipeline as P

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
KNN_KS = [5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 4000]
RF_DEPTHS = [2, 3, 4, 6, 8, 12, 16]
RF_EVERY = 2          # random forest trains on every 2nd daily origin (speed)
TRAIN_SAMPLE = 6000

D = M.Data()
FS = M.feature_sets(D.G)
rng = np.random.default_rng(0)
rows, preds = [], {}     # preds[(model, hp, year)] = (validation prediction)
truth = {}               # truth[year] = (y_val, lead_val, yp_baseline, last24_baseline)


def run(model_name, hp_name, hp, make, cols, every=1):
    for yr, tests in D.folds():
        first = tests[0]
        tr = D.train_origins(first, every)
        X_tr, y_tr = D.build(tr)
        assert X_tr.index.max() < first, "training target reaches the first validation origin"
        X_va, y_va = D.build(tests)
        if yr not in truth:
            lead = X_va["lead"].to_numpy()
            last24 = np.concatenate([np.tile(D.t.reindex(pd.date_range(o - pd.Timedelta(hours=24), periods=24, freq="h")).interpolate(limit_direction="both").to_numpy(), 14)[
                D.cache[o][1].notna().to_numpy()] for o in tests])
            truth[yr] = (y_va.to_numpy(), lead, X_va["yp_mean_sm7"].to_numpy(), last24)
        m = make(hp).fit(X_tr[cols], y_tr)
        # training error is measured on the SAME season (Aug 15 - Oct 15) of the training years, so it is
        # comparable with validation error (all-year rows would be harder, which hides overfitting)
        loc = X_tr.index.tz_convert(P.TZ)
        md = loc.month * 100 + loc.day
        in_season = np.flatnonzero((md >= 815) & (md <= 1015))
        pool = in_season if len(in_season) >= 2000 else np.arange(len(X_tr))
        idx = rng.choice(pool, size=min(TRAIN_SAMPLE, len(pool)), replace=False)
        p_tr, p_va = m.predict(X_tr[cols].iloc[idx]), m.predict(X_va[cols])
        preds[(model_name, str(hp), yr)] = p_va
        rows.append(dict(model=model_name, hyper_name=hp_name, hyper=hp, fold=yr, n_train=len(X_tr),
                         train_mae=M.mae(y_tr.iloc[idx], p_tr), val_mae=M.mae(y_va, p_va),
                         val_rmse=M.rmse(y_va, p_va), train_rmse=M.rmse(y_tr.iloc[idx], p_tr)))
        print(f"{model_name:6s} {hp_name}={hp!s:30s} fold {yr}: train {rows[-1]['train_mae']:.2f}  val {rows[-1]['val_mae']:.2f}", flush=True)


def best(df_rows, key="hyper"):
    d = pd.DataFrame(df_rows).groupby(key)["val_mae"].mean()
    return d.idxmin()


def table(model, hp_name=None):
    d = pd.DataFrame(rows)
    d = d[d.model == model]
    return d if hp_name is None else d[d.hyper_name == hp_name]


t0 = time.time()
# ---- ridge: which feature groups, then which alpha ----
for name, cols in FS.items():
    run("ridge", "features", name, lambda _: M.make_ridge(1.0), cols)
ridge_feat = best(table("ridge", "features").to_dict("records"))
for a in RIDGE_ALPHAS:
    run("ridge", "alpha", a, M.make_ridge, FS[ridge_feat])
ridge_alpha = best(table("ridge", "alpha").to_dict("records"))
ridge_alpha = next(a for a in RIDGE_ALPHAS if float(a) == float(ridge_alpha))   # back to the original type (dict key)

# ---- KNN: which compact feature set (at k=200), then which k ----
for name, cols in M.KNN_SETS.items():
    run("knn", "features", name, lambda _: M.make_knn(200), cols)
knn_feat = best(table("knn", "features").to_dict("records"))
for k in KNN_KS:
    run("knn", "k", k, M.make_knn, M.KNN_SETS[knn_feat])
knn_k = best(table("knn", "k").to_dict("records"))
knn_k = next(k for k in KNN_KS if k == int(knn_k))

# ---- random forest: depth (model complexity) on the lean base feature set ----
rf_cols = FS["base (RDU + year-prior)"]
for dpt in RF_DEPTHS:
    run("rf", "max_depth", dpt, M.make_rf, rf_cols, every=RF_EVERY)
rf_depth = best(table("rf", "max_depth").to_dict("records"))
rf_depth = next(d for d in RF_DEPTHS if d == int(rf_depth))
print(f"selection done in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)

df = pd.DataFrame(rows)
df.to_csv("results/cv_all_runs.csv", index=False)
for model, hp in [("ridge", "features"), ("ridge", "alpha"), ("knn", "features"), ("knn", "k"), ("rf", "max_depth")]:
    d = df[(df.model == model) & (df.hyper_name == hp)]
    g = d.groupby("hyper", sort=False)[["train_mae", "val_mae", "val_rmse", "train_rmse"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g.to_csv(f"results/cv_{model}_{hp}.csv")

# ---- compare the three tuned models and the baselines on the same folds ----
chosen = {"ridge": dict(features=ridge_feat, alpha=float(ridge_alpha)),
          "knn": dict(features=knn_feat, k=int(knn_k)),
          "rf": dict(features="base (RDU + year-prior)", max_depth=int(rf_depth), min_samples_leaf=20, n_estimators=60)}
best_hp = {"ridge": ridge_alpha, "knn": knn_k, "rf": rf_depth}
comp, lead_rows = [], []
for yr, (y, lead, yp, last24) in truth.items():
    series = {"Ridge linear regression": preds[("ridge", str(ridge_alpha), yr)], "KNN": preds[("knn", str(knn_k), yr)],
              "Random forest": preds[("rf", str(rf_depth), yr)], "Baseline: same hour, last 2 yrs": yp,
              "Baseline: repeat last 24 h": last24}
    for name, p in series.items():
        comp.append(dict(model=name, fold=yr, mae=M.mae(y, p), rmse=M.rmse(y, p)))
        for d_ in range(14):
            s = (lead >= 24 * d_) & (lead < 24 * (d_ + 1))
            lead_rows.append(dict(model=name, fold=yr, lead_day=d_ + 1, abs_err_sum=np.abs(y[s] - p[s]).sum(), n=int(s.sum())))
comp = pd.DataFrame(comp)
summary = comp.groupby("model")[["mae", "rmse"]].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.sort_values("mae_mean")
summary.to_csv("results/cv_model_comparison.csv")
comp.to_csv("results/cv_model_comparison_by_fold.csv", index=False)
lead = pd.DataFrame(lead_rows).groupby(["model", "lead_day"])[["abs_err_sum", "n"]].sum()
(lead.abs_err_sum / lead.n).rename("mae").reset_index().to_csv("results/cv_error_by_lead_day.csv", index=False)

# the better of KNN vs random forest becomes the "other" model
other = "knn" if summary.loc["KNN", "mae_mean"] <= summary.loc["Random forest", "mae_mean"] else "rf"
chosen["other_model"] = other
json.dump(chosen, open("results/chosen_models.json", "w"), indent=2)
print(summary.round(3).to_string())
print(json.dumps(chosen, indent=2))
