"""Rolling-origin backtest: which data groups and which "other" model to use.

Task replayed: at local midnight Sep 17 predict the next 336 hourly RDU temperatures using only
data before that moment.  Repeated for Sep 17 of 2022-2025; each fit uses only origins whose
14-day target window ends before the test origin (expanding window).

Run from repo root after `bash experiments/fetch_alt_data.sh`:
    python -W ignore experiments/backtest.py
Leakage guards live in pipeline.py and are verified by test_no_leakage.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline as P

TEST_YEARS = [2022, 2023, 2024, 2025]
S = P.load_sources()
O, Y, G = P.build_features(S)
t = S["rdu"]["temperature"]
origins = pd.date_range("2021-01-09", "2026-09-16", freq="D", tz=P.TZ).tz_convert("UTC")
cache = {o: P.make_features(O, Y, t, o) for o in origins}
for yr in TEST_YEARS:
    o = pd.Timestamp(f"{yr}-09-17", tz=P.TZ).tz_convert("UTC")
    cache[o] = P.make_features(O, Y, t, o)


def build(os_):
    X, y = map(pd.concat, zip(*(cache[o] for o in os_)))
    keep = y.notna()
    return X[keep], y[keep]


base = G["rdu"] + G["yearprior"] + P.CAL
SETS = {"A rdu only": G["rdu"] + P.CAL, "B + year-prior": base}
for g in ["corridor", "wind", "land", "sst", "indices"]:
    SETS[f"B + {g}"] = base + G[g]
SETS["all groups"] = base + sum((G[g] for g in ["corridor", "wind", "land", "sst", "indices"]), [])

KNN_BASE = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead", "rdu_t_24", "rdu_td_24", "rdu_slp_d24", "yp_mean_sm7"]
KNN_SETS = {
    "compact": KNN_BASE,
    "compact + dca": KNN_BASE + ["dca_minus_rdu_t24"],
    "compact + soil": KNN_BASE + ["land_rdu_sm_shallow_7d"],
    "compact + indices": KNN_BASE + ["idx_nao_7d", "idx_ao_7d"],
    "compact + sst": KNN_BASE + ["sst_gulfstream"],
}


def pipe(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


rows = []
for yr in TEST_YEARS:
    T = pd.Timestamp(f"{yr}-09-17", tz=P.TZ).tz_convert("UTC")
    X_tr, y_tr = build([o for o in origins if o <= T - pd.Timedelta(hours=P.H)])
    assert X_tr.index.max() < T, "training target reaches the test origin"
    X_te, y_te = cache[T]
    keep = y_te.notna()
    X_te, y_te = X_te[keep], y_te[keep].values
    lead = X_te["lead"].values

    def record(feats, model, pred):
        e = np.abs(pred - y_te)
        rows.append(dict(year=yr, features=feats, model=model, MAE=e.mean(), day1=e[lead < 24].mean(),
                         day2_3=e[(lead >= 24) & (lead < 72)].mean(), day4_14=e[lead >= 72].mean()))
        print(yr, feats, model, round(e.mean(), 2), flush=True)

    record("baseline", "same hour last 2 yrs", X_te["yp_mean_sm7"].values)
    last24 = t.reindex(pd.date_range(T - pd.Timedelta(hours=24), periods=24, freq="h")).values
    record("baseline", "repeat last 24 h", np.tile(last24, 14)[keep.values])
    preds = {}
    for name, cols in SETS.items():
        p = pipe(Ridge(alpha=1.0)).fit(X_tr[cols], y_tr).predict(X_te[cols])
        record(name, "LinReg (ridge)", p)
        preds[name] = p
    for name, cols in KNN_SETS.items():
        p = pipe(KNeighborsRegressor(300, n_jobs=-1)).fit(X_tr[cols], y_tr).predict(X_te[cols])
        record(name, "KNN k=300", p)
        preds[name] = p
    record("B + year-prior / compact", "average of ridge + KNN", (preds["B + year-prior"] + preds["compact"]) / 2)

res = pd.DataFrame(rows)
Path("experiments/results").mkdir(exist_ok=True)
res.to_csv("experiments/results/backtest_results.csv", index=False)
summary = res.groupby(["features", "model"])[["MAE", "day1", "day2_3", "day4_14"]].mean()
by_year = res.pivot_table(index=["features", "model"], columns="year", values="MAE")
print("\nMAE (deg C), mean of Sep 17-30 in", TEST_YEARS)
print(summary.join(by_year).sort_values("MAE").round(2).to_string())
