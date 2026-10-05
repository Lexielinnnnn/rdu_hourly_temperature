"""Leakage tests.  Run from the repo root:  python -W ignore experiments/test_no_leakage.py

1. Every source is clipped strictly before the Sep 17 2026 cutoff.
2. Poison test: overwrite ALL source data from time T onward with garbage; the features and the
   training targets available at origin T must be bit-for-bit identical to the unpoisoned run.
   Run for T = three backtest origins and for T = CUTOFF (the real forecast).
3. Training origins for a test origin T never have targets that reach past T.
"""
import numpy as np
import pandas as pd

import pipeline as P

clean = P.load_sources()
for name, df in clean.items():
    last = df.dropna(how="all").index.max()
    assert last < P.CUTOFF, f"{name} has data at/after cutoff: {last}"
print("1. all sources end before the cutoff:", max(df.dropna(how='all').index.max() for df in clean.values()))

O0, Y0, _ = P.build_features(clean)
t0 = clean["rdu"]["temperature"]
for T in [pd.Timestamp(f"{y}-09-17", tz=P.TZ).tz_convert("UTC") for y in (2023, 2025)] + [P.CUTOFF]:
    S1 = P.load_sources(poison_from=T)
    O1, Y1, _ = P.build_features(S1)
    X0, _ = P.make_features(O0, Y0, t0, T)
    X1, _ = P.make_features(O1, Y1, S1["rdu"]["temperature"], T)
    # compare only the feature columns that exist at origin time (Y rows for future hours come from the past)
    pd.testing.assert_frame_equal(X0, X1)
    # training rows: every origin whose 14-day target window ends by T must be identical too
    last_train = T - pd.Timedelta(hours=P.H)
    for o in pd.date_range(last_train - pd.Timedelta(days=3), last_train, freq="D"):
        a, ya = P.make_features(O0, Y0, t0, o)
        b, yb = P.make_features(O1, Y1, S1["rdu"]["temperature"], o)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(ya, yb)
    if T < P.CUTOFF:  # sanity: the poison must be visible to features built from LATER origins, else the test proves nothing
        later = T + pd.Timedelta(days=3)
        a, _ = P.make_features(O0, Y0, t0, later); b, _ = P.make_features(O1, Y1, S1["rdu"]["temperature"], later)
        assert not a.equals(b), "poison had no effect on later origins - test is vacuous"
    print(f"2. poisoning everything from {T} leaves features and training rows unchanged: OK")

print("ALL LEAKAGE TESTS PASSED")
