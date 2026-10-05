"""Lexie's random forest (from random_forest.ipynb, commit 854e0d9) with ONE fix.

Fix: the pushed notebook built its training forecasts at UTC midnight (8pm Raleigh time in September), but the real
forecast starts at midnight Raleigh time.  In training "lead hour 0" therefore meant 8pm and the "last hour" features
described the evening.  Here the training forecasts start at local midnight (`align="local"`), like the real one.
`align="utc"` reproduces the pushed behavior so the effect can be measured (see cross_validation.py).

Features, hyperparameters and everything else are unchanged from the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from rdu_data import HORIZON, TZ, WEATHER, local_midnights, utc_midnights


def make_features(data, origin):
    """20 features for the 336 hours starting at `origin`, from observations strictly before `origin`."""
    past = data.loc[(data["DATE"] >= origin - pd.Timedelta(days=7)) & (data["DATE"] < origin), WEATHER]
    recent = past.loc[past.index >= origin - pd.Timedelta(hours=24)]
    features = {}
    for column in WEATHER:
        features[f"{column}_last_hour"] = past[column].reindex([origin - pd.Timedelta(hours=1)]).iloc[0]
        features[f"{column}_mean_24h"] = recent[column].mean()
        features[f"{column}_mean_7d"] = past[column].mean()
    hours = pd.date_range(origin, periods=HORIZON, freq="h")
    local_hours = hours.tz_convert(TZ)
    X = pd.DataFrame(features, index=hours)
    X["lead_hour"] = np.arange(HORIZON)
    X["hour_sin"] = np.sin(2 * np.pi * local_hours.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * local_hours.hour / 24)
    X["day_sin"] = np.sin(2 * np.pi * local_hours.dayofyear / 365.25)
    X["day_cos"] = np.cos(2 * np.pi * local_hours.dayofyear / 365.25)
    return X


def training_origins(data, end, align="local", every=1):
    """Daily forecast origins whose whole 14-day target window ends before `end`."""
    first = data.index.min() + pd.Timedelta(days=7)
    last = end - pd.Timedelta(hours=HORIZON)
    origins = local_midnights(first, last) if align == "local" else utc_midnights(first, last)
    return origins[::every]


def make_training_data(data, end, align="local", every=1, cache=None):
    X_parts, y_parts = [], []
    for origin in training_origins(data, end, align, every):
        X = cache[origin] if cache is not None and origin in cache else make_features(data, origin)
        if cache is not None:
            cache[origin] = X
        y = data["temperature"].reindex(X.index)
        valid = y.notna()
        X_parts.append(X.loc[valid])
        y_parts.append(y.loc[valid])
    X, y = pd.concat(X_parts), pd.concat(y_parts)          # index = the hour each row predicts (not unique: forecasts overlap)
    assert X.index.max() < end, "a training target reaches the forecast start"
    return X, y


def make_model(max_depth=16, n_estimators=100, max_samples=0.8, min_samples_leaf=20, max_features=0.8, seed=42):
    """Defaults are the notebook's settings.  cross_validation.py uses fewer/cheaper trees for speed."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                              max_features=max_features, max_samples=max_samples, random_state=seed, n_jobs=-1),
    )
