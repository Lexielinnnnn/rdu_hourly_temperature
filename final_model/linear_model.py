"""The two-stage linear model (from linear_regression.ipynb, commit 71c7270).

Stage 1: plain linear regression on day-of-year x hour-of-day Fourier terms = the "normal" temperature.
Stage 2: ridge regression predicting the departure from normal at each lead hour, from the weather at the forecast origin,
         with each feature faded out over lead time (exp(-lead/tau)).
Prediction = normal + predicted departure.

The logic is copied from the notebook.  Changes: (1) the data is passed in instead of being a global, (2) `predict_many`
and `set_alpha` were added so cross-validation can refit/score quickly, (3) the training design matrix is kept so
training error can be measured.  Training forecasts already start at LOCAL midnight (no alignment problem here).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rdu_data import CLIM_COLS, HORIZON, TZ

FEATURE_GROUPS = {
    "temp": ["t_last", "t_6h", "t_24h", "t_72h", "t_7d"],      # temperature departure at several windows
    "dewpoint": ["td_24h"],                                   # moisture departure (air mass)
    "pressure": ["p_anom", "p_tend_6h", "p_tend_24h"],        # pressure level and trend (approaching systems)
}
DECAY_TAUS = [3, 12, 48, 168]   # hours: how fast each signal fades with lead time


def clim_basis(ts_utc, k_doy=3, k_hr=3):
    t = ts_utc.tz_convert(TZ)
    doy = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    hr = 2 * np.pi * t.hour.values / 24
    d = [np.ones(len(t))] + [f(k * doy) for k in range(1, k_doy + 1) for f in (np.sin, np.cos)]
    h = [np.ones(len(t))] + [f(k * hr) for k in range(1, k_hr + 1) for f in (np.sin, np.cos)]
    return np.column_stack([a * b for a in d for b in h])[:, 1:]


class Climatology:
    def __init__(self, hourly, col):
        self.hourly, self.col = hourly, col

    def fit(self, end):
        y = self.hourly.loc[self.hourly.index < end, self.col].dropna()
        self.model = LinearRegression().fit(clim_basis(y.index), y.values)
        return self

    def predict(self, ts_utc):
        return self.model.predict(clim_basis(ts_utc))


def origin_features(hourly, A, origins, groups):
    ta = A["temperature"]
    pres = hourly["sea_level_pressure"].ffill(limit=3)
    f = pd.DataFrame(index=A.index)
    f["t_last"] = ta.ffill(limit=3)
    f["t_6h"] = ta.rolling(6, min_periods=3).mean()
    f["t_24h"] = ta.rolling(24, min_periods=12).mean()
    f["t_72h"] = ta.rolling(72, min_periods=36).mean()
    f["t_7d"] = ta.rolling(168, min_periods=84).mean()
    f["td_24h"] = A["dew_point_temperature"].rolling(24, min_periods=12).mean()
    f["p_anom"] = A["sea_level_pressure"].ffill(limit=3)
    f["p_tend_6h"] = pres - pres.shift(6)
    f["p_tend_24h"] = pres - pres.shift(24)
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    F = f[cols].reindex(origins - pd.Timedelta(hours=1)).fillna(0.0)     # row (origin - 1h) = everything observed before the origin
    return F.values, cols


def design(F, origins, cols):
    lead = np.arange(HORIZON)
    decay = np.column_stack([np.exp(-lead / tau) for tau in DECAY_TAUS])
    X = np.einsum("ok,lm->olkm", F, decay).reshape(len(F) * HORIZON, -1)
    names = [f"{c} x exp(-L/{tau}h)" for c in cols for tau in DECAY_TAUS]
    short = [i for i, c in enumerate(cols) if c in ("t_last", "t_6h", "t_24h")]
    if short:
        target_hr = np.array([pd.date_range(o, periods=HORIZON, freq="h").tz_convert(TZ).hour.values for o in origins])
        hr = 2 * np.pi * target_hr / 24
        H = np.stack([np.exp(-lead / 24) * f(k * hr) for k in (1, 2) for f in (np.sin, np.cos)], axis=-1)
        X2 = np.einsum("ok,olm->olkm", F[:, short], H).reshape(len(F) * HORIZON, -1)
        X = np.hstack([X, X2])
        names += [f"{cols[i]} x hour-{k}{fn}" for i in short for k in (1, 2) for fn in ("sin", "cos")]
    return X, names


class LinearForecaster:
    """Climatology regression + ridge regression on departures.  Trained only on data before `end`."""

    def __init__(self, hourly, groups=("temp",), alpha=1.0):
        self.hourly, self.groups, self.alpha = hourly, list(groups), alpha

    def fit(self, end):
        end = end.tz_convert("UTC")
        self.end = end
        self.clims = {c: Climatology(self.hourly, c).fit(end) for c in CLIM_COLS}
        self.A = pd.DataFrame({c: self.hourly[c].values - m.predict(self.hourly.index) for c, m in self.clims.items()},
                              index=self.hourly.index)
        # one training forecast per day at LOCAL midnight (same as the real forecast), all 336 targets before `end`
        first = (self.hourly.index.min().tz_convert(TZ) + pd.Timedelta(days=8)).normalize()
        last = (end - pd.Timedelta(hours=HORIZON)).tz_convert(TZ)
        origins = pd.date_range(first, last, freq="D", tz=TZ).tz_convert("UTC")
        F, cols = origin_features(self.hourly, self.A, origins, self.groups)
        X, self.feature_names = design(F, origins, cols)
        targets = origins.repeat(HORIZON) + pd.to_timedelta(np.tile(np.arange(HORIZON), len(origins)), unit="h")
        y = self.A["temperature"].reindex(targets).values
        ok = ~np.isnan(y)
        self._X, self._y, self._targets = X[ok], y[ok], targets[ok]
        assert self._targets.max() < end, "a training target reaches the forecast start"
        self.n_train = int(ok.sum())
        return self.set_alpha(self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(self._X, self._y)
        return self

    def predict_many(self, origins):
        """(n_origins x 336) predictions for forecasts issued at the given local-midnight origins."""
        o = pd.DatetimeIndex([x.tz_convert("UTC") for x in origins])
        F, cols = origin_features(self.hourly, self.A, o, self.groups)
        X, _ = design(F, o, cols)
        hours = pd.DatetimeIndex(np.concatenate([pd.date_range(x, periods=HORIZON, freq="h").values for x in o])).tz_localize("UTC")
        return (self.clims["temperature"].predict(hours) + self.model.predict(X)).reshape(len(o), HORIZON)

    def predict(self, origin):
        return self.predict_many([origin])[0]

    def climatology_only(self, origin):
        hours = pd.date_range(origin.tz_convert("UTC"), periods=HORIZON, freq="h")
        return self.clims["temperature"].predict(hours)

    def train_error_sample(self, season=(815, 1015), n=6000, seed=0):
        """MAE on training rows from the same Aug 15 - Oct 15 season as the validation forecasts."""
        loc = self._targets.tz_convert(TZ)
        md = loc.month * 100 + loc.day
        pool = np.flatnonzero((md >= season[0]) & (md <= season[1]))
        idx = np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False)
        return float(np.mean(np.abs(self.model.predict(self._X[idx]) - self._y[idx])))


def climatology_baseline(temp_local, origin):
    """Mean temperature for the same local hour within +/-7 days of the date, prior years only."""
    past = temp_local[temp_local.index < origin].dropna()
    sums, counts = np.zeros((367, 24)), np.zeros((367, 24))
    np.add.at(sums, (past.index.dayofyear.values, past.index.hour.values), past.values)
    np.add.at(counts, (past.index.dayofyear.values, past.index.hour.values), 1)
    out = []
    for h in pd.date_range(origin, periods=HORIZON, freq="h", tz=TZ):
        days = (np.arange(h.dayofyear - 7, h.dayofyear + 8) - 1) % 366 + 1
        out.append(sums[days, h.hour].sum() / counts[days, h.hour].sum())
    return np.array(out)


def persistence_baseline(temp_local, origin):
    prev = temp_local.reindex(pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="h", tz=TZ))
    return np.tile(prev.interpolate(limit_direction="both").values, 14)
