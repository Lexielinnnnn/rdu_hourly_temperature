"""Shared pieces for model selection and the final forecast: features, models, time-ordered folds.

Everything here sits on top of pipeline.py, which clips every source at the Sep 17 2026 cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline as P

CV_YEARS = [2022, 2023, 2024, 2025]            # one fold per year
SEASON_START, SEASON_END, STEP = "08-15", "10-15", "3D"   # validation forecasts: every 3 days, Aug 15 - Oct 15
FIRST_ORIGIN = "2021-01-09"


class Data:
    """Loads all sources once and builds the feature matrix for every daily forecast origin."""

    def __init__(self):
        self.S = P.load_sources()
        self.O, self.Y, self.G = P.build_features(self.S)
        self.t = self.S["rdu"]["temperature"]
        # one forecast per day, issued at local midnight (the real forecast is issued the same way)
        self.origins = pd.date_range(FIRST_ORIGIN, "2026-09-17", freq="D", tz=P.TZ).tz_convert("UTC")
        self.cache = {o: P.make_features(self.O, self.Y, self.t, o) for o in self.origins}

    def build(self, origins):
        X, y = map(pd.concat, zip(*(self.cache[o] for o in origins)))
        keep = y.notna()
        return X[keep], y[keep]

    def train_origins(self, before, every=1):
        """Origins whose whole 14-day target window ends before `before` (no overlap with the test period)."""
        ok = [o for o in self.origins if o <= before - pd.Timedelta(hours=P.H)]
        return ok[::every]

    def folds(self):
        """Yield (year, test_origins).  Train on everything that ends before the season's first origin."""
        for yr in CV_YEARS:
            tests = pd.date_range(f"{yr}-{SEASON_START}", f"{yr}-{SEASON_END}", freq=STEP, tz=P.TZ).tz_convert("UTC")
            yield yr, tests


def feature_sets(G):
    base = G["rdu"] + G["yearprior"] + P.CAL
    return {
        "base (RDU + year-prior)": base,
        "base + soil/cloud": base + G["land"],
        "base + wind": base + G["wind"],
        "base + soil/cloud + wind": base + G["land"] + G["wind"],
        "base + stations": base + G["corridor"],
        "base + ocean temp": base + G["sst"],
        "base + climate indices": base + G["indices"],
    }


KNN_BASE = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead", "rdu_t_24", "rdu_td_24", "rdu_slp_d24", "yp_mean_sm7"]
KNN_SETS = {"compact": KNN_BASE, "compact + DC temperature": KNN_BASE + ["dca_minus_rdu_t24"]}


def make_ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def make_knn(k):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(k, n_jobs=-1))


def make_rf(max_depth):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(60, max_depth=max_depth, min_samples_leaf=20, max_features=0.6,
                                               max_samples=0.3, n_jobs=-1, random_state=0))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
