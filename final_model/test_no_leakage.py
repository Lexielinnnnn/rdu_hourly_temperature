"""Leakage tests for both final models.  Run from the repo root:  python -W ignore final_model/test_no_leakage.py

1. No observation at or after the Sep 17 2026 cutoff is loaded.
2. Poison test: every observation from forecast origin T onward is replaced by garbage (1000).  The features, the
   training data, the fitted model and the forecast made at T must be IDENTICAL to the unpoisoned run.
   Run for T = Sep 17 of 2023 and 2025, and for T = the real cutoff.
3. Sanity check: the poison DOES change a model that is allowed to see it (so the test is not vacuous).
4. Training forecasts for the random forest start at local midnight (the real forecast does too).
"""
import numpy as np
import pandas as pd

import linear_model as LM
import rdu_data as R
import rf_model as RF

clean = R.load()
assert clean["DATE"].max() < R.CUTOFF
print("1. last observation loaded:", clean["DATE"].max().tz_convert(R.TZ), "(cutoff", R.CUTOFF.tz_convert(R.TZ), ")")

origins = [pd.Timestamp(f"{y}-09-17", tz=R.TZ).tz_convert("UTC") for y in (2023, 2025)] + [R.CUTOFF]
for T in origins:
    bad = R.load(poison_from=T)
    # --- random forest ---
    pd.testing.assert_frame_equal(RF.make_features(clean, T), RF.make_features(bad, T))
    Xa, ya = RF.make_training_data(clean, T, every=45)
    Xb, yb = RF.make_training_data(bad, T, every=45)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)
    fa = RF.make_model(n_estimators=10, max_samples=0.3).fit(Xa, ya).predict(RF.make_features(clean, T))
    fb = RF.make_model(n_estimators=10, max_samples=0.3).fit(Xb, yb).predict(RF.make_features(bad, T))
    np.testing.assert_allclose(fa, fb, rtol=0, atol=1e-9)   # threads sum trees in a different order: 1e-15 noise
    # --- Burak's linear model ---
    la = LM.LinearForecaster(R.hourly_grid(clean)).fit(T)
    lb = LM.LinearForecaster(R.hourly_grid(bad)).fit(T)
    np.testing.assert_allclose(la.predict(T.tz_convert(R.TZ)), lb.predict(T.tz_convert(R.TZ)), rtol=0, atol=1e-9)
    print(f"2. poison from {T.tz_convert(R.TZ)}: random forest and linear model forecasts unchanged: OK")

# 3. the poison must matter to a model that is allowed to see it
T = origins[0]
bad = R.load(poison_from=T)
later = T + pd.Timedelta(days=20)
a = LM.LinearForecaster(R.hourly_grid(clean)).fit(later).predict(later.tz_convert(R.TZ))
b = LM.LinearForecaster(R.hourly_grid(bad)).fit(later).predict(later.tz_convert(R.TZ))
assert not np.allclose(a, b), "poison had no effect even when allowed - test would be vacuous"
print("3. poison changes a model that may see it (test is not vacuous): OK")

# 4. alignment
o = RF.training_origins(clean, R.CUTOFF, align="local")
assert set(o.tz_convert(R.TZ).hour) == {0}, "training forecasts must start at local midnight"
print("4. random forest training forecasts all start at local midnight (hour 0 Raleigh time): OK")
print("ALL LEAKAGE TESTS PASSED")
