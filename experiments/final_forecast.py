"""Train the chosen models on ALL data before the Sep 17 2026 cutoff and forecast Sep 17-30 2026.

This script never reads the real Sep 17-30 observations (pipeline.py clips every source at the cutoff).
Outputs (repo root): predictions_ridge.csv, predictions_knn.csv, predictions_rf.csv
         results/: ridge_coefficients.csv, rf_importances.csv, forecast_feature_check.txt
Run from repo root after model_selection.py:  python -W ignore experiments/final_forecast.py
"""
import json

import numpy as np
import pandas as pd

import modeling as M
import pipeline as P

chosen = json.load(open("results/chosen_models.json"))
D = M.Data()
FS = M.feature_sets(D.G)
X_fc, _ = D.cache[P.CUTOFF]          # features for the 336 forecast hours, built from data before the cutoff only
hours_local = X_fc.index.tz_convert(P.TZ)


def fit_and_predict(name, make, cols, every=1):
    tr = D.train_origins(P.CUTOFF, every)
    X_tr, y_tr = D.build(tr)
    assert X_tr.index.max() < P.CUTOFF, "training target at/after the cutoff"
    m = make().fit(X_tr[cols], y_tr)
    pd.DataFrame({"hour_local": hours_local, "predicted_temperature": m.predict(X_fc[cols])}).to_csv(
        f"predictions_{name}.csv", index=False)
    print(f"{name}: trained on {len(X_tr):,} rows ending {X_tr.index.max()}; wrote predictions_{name}.csv")
    return m


r, k, f = chosen["ridge"], chosen["knn"], chosen["rf"]
ridge_cols = FS[r["features"]]
ridge = fit_and_predict("ridge", lambda: M.make_ridge(r["alpha"]), ridge_cols)
knn = fit_and_predict("knn", lambda: M.make_knn(k["k"]), M.KNN_SETS[k["features"]])
rf_cols = FS[f["features"]]
rf = fit_and_predict("rf", lambda: M.make_rf(f["max_depth"]), rf_cols, every=2)

pd.Series(ridge[-1].coef_, index=ridge_cols).rename("standardized_coef").to_csv("results/ridge_coefficients.csv")
pd.Series(rf[-1].feature_importances_, index=rf_cols).rename("importance").sort_values(ascending=False).to_csv("results/rf_importances.csv")

nan = X_fc[sorted(set(ridge_cols) | set(M.KNN_SETS[k["features"]]))].isna().any()
with open("results/forecast_feature_check.txt", "w") as fh:
    fh.write(f"forecast rows: {len(X_fc)}; features built from data before {P.CUTOFF}\n")
    fh.write("features with missing values at forecast time (median-imputed): " + (", ".join(nan[nan].index) or "none") + "\n")
print(open("results/forecast_feature_check.txt").read())
