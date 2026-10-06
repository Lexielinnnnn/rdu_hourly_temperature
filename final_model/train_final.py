"""Train the two final models on ALL data before the Sep 17 2026 cutoff and forecast Sep 17-30 2026.

This script never reads the real Sep 17-30 observations (rdu_data.load() drops everything at/after the cutoff).
Hyperparameters come from cross_validation.py (results/chosen.json); the random forest uses the notebook's full
settings (100 trees, 80% bootstrap) at the CV-chosen depth.
Outputs: predictions/{rf,linear}_predictions.csv, results/{rf_importances,linear_coefficients}.csv
Run from repo root:  python -W ignore final_model/train_final.py
"""
import json

import pandas as pd

import linear_model as LM
import rdu_data as R
import rf_model as RF

chosen = json.load(open("final_model/results/chosen.json"))
data = R.load()
hourly = R.hourly_grid(data)
hours_local = pd.date_range(R.CUTOFF, periods=R.HORIZON, freq="h").tz_convert(R.TZ)

# ---- Random forest, training forecasts at local midnight ----
X, y = RF.make_training_data(data, R.CUTOFF, align="local", every=1)
rf = RF.make_model(max_depth=chosen["rf"]["max_depth"]).fit(X, y)
Xf = RF.make_features(data, R.CUTOFF)
pd.DataFrame({"hour_local": hours_local, "predicted_temperature": rf.predict(Xf)}).to_csv(
    "final_model/predictions/rf_predictions.csv", index=False)
pd.Series(rf[-1].feature_importances_, index=Xf.columns, name="importance").sort_values(ascending=False).to_csv(
    "final_model/results/rf_importances.csv")
print(f"random forest: depth {chosen['rf']['max_depth']}, {len(X):,} training rows, last target {X.index.max().tz_convert(R.TZ)}")

# ---- Two-stage linear model ----
lin = LM.LinearForecaster(hourly, chosen["linear"]["groups"], alpha=chosen["linear"]["alpha"]).fit(R.CUTOFF)
pd.DataFrame({"hour_local": hours_local, "predicted_temperature": lin.predict(R.CUTOFF)}).to_csv(
    "final_model/predictions/linear_predictions.csv", index=False)
pd.Series(lin.model[-1].coef_, index=lin.feature_names, name="standardized_coef").to_csv("final_model/results/linear_coefficients.csv")
print(f"linear model: groups {chosen['linear']['groups']}, alpha {chosen['linear']['alpha']}, {lin.n_train:,} training rows, "
      f"last target {lin._targets.max().tz_convert(R.TZ)}")
