"""The MOS model: ECMWF forecast + ridge-regression correction (from nwp_mos.ipynb).

Model Output Statistics (MOS): a linear regression that corrects a weather model's raw forecast for one station.
  - Input weather model: ECMWF IFS HRES 12 UTC runs (nwp_data/ifs_12z_rdu.csv, from fetch_ifs_runs.py).
  - A forecast issued at local midnight uses the PREVIOUS day's 12 UTC run (8am EDT, published mid-afternoon),
    i.e. a run that started >= 12 h before the forecast is issued.
  - Everything is predicted as a departure from the Fourier climatology (same climatology as linear_model.py).
  - Hours beyond the end of the ECMWF run fall back to a station-only ridge model.

The logic is copied from the notebook.  The only change: the data is held in a `Data` object instead of notebook
globals, so other scripts (presentation/make_data.py) can import it.  Run from the repo root.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "America/New_York"
HORIZON = 14 * 24
CUTOFF = pd.Timestamp("2026-09-17 00:00", tz=TZ)
RUN_HOUR_UTC = 12                          # which daily ECMWF run to use
MIN_RUN_AGE = pd.Timedelta(hours=12)       # the run must start >= 12 h before the forecast is issued
LEAD = np.arange(HORIZON)
STATION_TAUS = [3, 12, 48]


class Data:
    """Observations (clipped at the cutoff) and ECMWF runs (only runs started >= 12 h before the cutoff)."""

    def __init__(self, obs_csv="cleaned_data/rdu_51_clean.csv", nwp_csv="nwp_data/ifs_12z_rdu.csv"):
        raw = pd.read_csv(obs_csv)
        raw["hour"] = pd.to_datetime(raw["DATE"], utc=True).dt.floor("h")
        raw = raw.drop_duplicates("hour").set_index("hour").sort_index()
        grid = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
        obs = raw.reindex(grid)["temperature"]
        self.obs = obs.loc[obs.index < CUTOFF]                                       # the leakage guard
        self.temp_local = pd.Series(self.obs.values, index=self.obs.index.tz_convert(TZ))

        nwp = pd.read_csv(nwp_csv, parse_dates=["init_utc", "valid_utc"])
        nwp = nwp[nwp["init_utc"].dt.hour == RUN_HOUR_UTC]
        assert nwp["init_utc"].max() <= CUTOFF - MIN_RUN_AGE, "a run newer than the cutoff allows is in the file"
        self.nwp = nwp
        self.nwp_temp = nwp.set_index(["init_utc", "valid_utc"])["nwp_temp"].sort_index()
        self.available_runs = pd.DatetimeIndex(nwp["init_utc"].unique()).sort_values()

        # ECMWF's recent bias, known at any hour: mean error over hours already observed (leads <= 48 h).
        # Every run contributing to an hour started before that hour, so nothing here comes from the future.
        short = nwp[nwp["lead_h"] <= 48].copy()
        short["err"] = short["nwp_temp"].values - self.obs.reindex(short["valid_utc"]).values
        self.err_by_hour = short.groupby("valid_utc")["err"].mean().reindex(self.obs.index)
        self.recent_bias = pd.DataFrame({
            "bias_7d": self.err_by_hour.rolling(168, min_periods=48).mean(),
            "bias_30d": self.err_by_hour.rolling(720, min_periods=168).mean(),
        })

    def run_for(self, origin):
        """Latest 12 UTC run that started at least MIN_RUN_AGE before the forecast origin."""
        latest_ok = origin.tz_convert("UTC") - MIN_RUN_AGE
        run = latest_ok.normalize() + pd.Timedelta(hours=RUN_HOUR_UTC)
        if run > latest_ok:
            run -= pd.Timedelta(days=1)
        return run if run in self.available_runs else None

    def nwp_path(self, origin, offsets):
        """ECMWF temperature from the run for `origin`, at origin + offsets (hours). NaN where not covered."""
        run = self.run_for(origin)
        times = origin.tz_convert("UTC") + pd.to_timedelta(offsets, unit="h")
        if run is None:
            return np.full(len(offsets), np.nan)
        return self.nwp_temp.reindex(pd.MultiIndex.from_arrays([np.repeat(run, len(times)), times])).values


def clim_basis(ts_utc, k_doy=3, k_hr=3):
    t = ts_utc.tz_convert(TZ)
    doy = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    hr = 2 * np.pi * t.hour.values / 24
    d = [np.ones(len(t))] + [f(k * doy) for k in range(1, k_doy + 1) for f in (np.sin, np.cos)]
    h = [np.ones(len(t))] + [f(k * hr) for k in range(1, k_hr + 1) for f in (np.sin, np.cos)]
    return np.column_stack([a * b for a in d for b in h])[:, 1:]


class Climatology:
    def __init__(self, D):
        self.D = D

    def fit(self, end):
        y = self.D.obs.loc[self.D.obs.index < end].dropna()
        self.model = LinearRegression().fit(clim_basis(y.index), y.values)
        return self

    def predict(self, ts_utc):
        return self.model.predict(clim_basis(ts_utc))


def station_features(anom):
    """Recent observed anomaly at several windows, read at the hour before each origin."""
    return pd.DataFrame({
        "t_last": anom.ffill(limit=3),
        "t_6h": anom.rolling(6, min_periods=3).mean(),
        "t_24h": anom.rolling(24, min_periods=12).mean(),
        "t_72h": anom.rolling(72, min_periods=36).mean(),
    })


def _before(origins):
    return [o.tz_convert("UTC") - pd.Timedelta(hours=1) for o in origins]


def mos_design(D, origins, clim, st_feats):
    """Rows = (origin, lead hour). Returns X, ECMWF-available mask, climatology values and feature names."""
    n = len(origins)
    hours_utc = [o.tz_convert("UTC") + pd.to_timedelta(LEAD, unit="h") for o in origins]
    clim_vals = np.array([clim.predict(h) for h in hours_utc])
    nwp_vals = np.array([D.nwp_path(o, LEAD) for o in origins])
    nwp_anom = nwp_vals - clim_vals

    obs_prev = D.obs.reindex(_before(origins)).values                     # current ECMWF error at 11pm
    nwp_prev = np.array([D.nwp_path(o, [-1])[0] for o in origins])
    err_now = np.nan_to_num(obs_prev - nwp_prev)

    F = st_feats.reindex(_before(origins)).fillna(0).values
    target_hr = np.array([pd.date_range(o, periods=HORIZON, freq="h", tz=TZ).hour.values for o in origins])
    hr = 2 * np.pi * target_hr / 24
    lead_day = np.broadcast_to(LEAD // 24, (n, HORIZON))

    cols, names = [], []
    for d in range(14):                                          # trust in ECMWF, by lead day
        cols.append(nwp_anom * (lead_day == d)); names.append(f"nwp_anom x day{d + 1}")
    for k in (1, 2):                                             # ECMWF diurnal-cycle correction
        for f, fn in ((np.sin, "sin"), (np.cos, "cos")):
            cols.append(nwp_anom * f(k * hr)); names.append(f"nwp_anom x hour{k}{fn}")
            cols.append(f(k * hr)); names.append(f"hour{k}{fn} (bias)")
    for tau in (6, 24):                                          # current ECMWF error, fading
        cols.append(err_now[:, None] * np.exp(-LEAD / tau)[None, :]); names.append(f"err_now x exp(-L/{tau}h)")
    B = D.recent_bias.reindex(_before(origins)).fillna(0).values
    for j, c in enumerate(D.recent_bias.columns):               # ECMWF's recent bias
        cols.append(B[:, j:j + 1]); names.append(c)
    for j, c in enumerate(st_feats.columns):                    # latest observations, fading
        for tau in STATION_TAUS:
            cols.append(F[:, j:j + 1] * np.exp(-LEAD / tau)[None, :]); names.append(f"{c} x exp(-L/{tau}h)")

    X = np.stack([np.broadcast_to(c, (n, HORIZON)) for c in cols], axis=-1).reshape(n * HORIZON, -1)
    has_nwp = ~np.isnan(nwp_anom).ravel()
    return np.nan_to_num(X), has_nwp, clim_vals.ravel(), names


def daily_origins(first, end):
    """Local-midnight origins whose 336 target hours all end before `end`."""
    last = (end.tz_convert("UTC") - pd.Timedelta(hours=HORIZON)).tz_convert(TZ)
    return pd.date_range(first.tz_convert(TZ).normalize(), last, freq="D", tz=TZ)


def targets(origins, anom):
    t = pd.DatetimeIndex(np.concatenate([o.tz_convert("UTC") + pd.to_timedelta(LEAD, unit="h") for o in origins]))
    return anom.reindex(t).values, t


class StationLinear:
    """Climatology + ridge on recent observed anomalies x lead-time decay (fallback past the ECMWF run)."""
    TAUS = [3, 12, 48, 168]

    def __init__(self, D):
        self.D = D

    def fit(self, end, clim, anom):
        self.clim = clim
        self.feats = station_features(anom).assign(t_7d=anom.rolling(168, min_periods=84).mean())
        origins = daily_origins(self.D.obs.index.min() + pd.Timedelta(days=8), end)
        y, t = targets(origins, anom)
        ok = ~np.isnan(y) & (t < end)
        self.model = make_pipeline(StandardScaler(), Ridge(1.0)).fit(self._X(origins)[ok], y[ok])
        return self

    def _X(self, origins):
        F = self.feats.reindex(_before(origins)).fillna(0).values
        dec = np.column_stack([np.exp(-LEAD / t) for t in self.TAUS])
        return np.einsum("ok,lm->olkm", F, dec).reshape(len(origins) * HORIZON, -1)

    def predict(self, origin):
        hours = origin.tz_convert("UTC") + pd.to_timedelta(LEAD, unit="h")
        return self.clim.predict(hours) + self.model.predict(self._X([origin]))


class MOS:
    def __init__(self, D):
        self.D = D

    def fit(self, end):
        D, end = self.D, end.tz_convert("UTC")
        self.clim = Climatology(D).fit(end)
        anom = D.obs - self.clim.predict(D.obs.index)
        self.st_feats = station_features(anom)
        self.fallback = StationLinear(D).fit(end, self.clim, anom)

        origins = [o for o in daily_origins(D.available_runs.min() + pd.Timedelta(days=1), end) if D.run_for(o) is not None]
        X, has_nwp, _, self.names = mos_design(D, origins, self.clim, self.st_feats)
        y, t = targets(origins, anom)
        ok = has_nwp & ~np.isnan(y) & (t < end)                  # never train on hours at/after `end`
        self.model = make_pipeline(StandardScaler(), Ridge(1.0)).fit(X[ok], y[ok])
        self.n_train, self.n_runs = int(ok.sum()), len(origins)
        return self

    def predict(self, origin):
        X, has_nwp, clim_vals, _ = mos_design(self.D, [origin], self.clim, self.st_feats)
        pred = self.fallback.predict(origin)
        pred[has_nwp] = clim_vals[has_nwp] + self.model.predict(X[has_nwp])
        return pred

    def raw_nwp(self, origin):
        """Uncorrected ECMWF, climatology where the run does not reach."""
        raw = self.D.nwp_path(origin, LEAD)
        clim = self.clim.predict(origin.tz_convert("UTC") + pd.to_timedelta(LEAD, unit="h"))
        return np.where(np.isnan(raw), clim, raw)

    def coefficients(self):
        """Ridge coefficients in original units (°C of correction per unit of each feature)."""
        scaler, ridge = self.model[0], self.model[-1]
        return pd.DataFrame({"feature": self.names, "coef_standardized": ridge.coef_,
                             "coef_original_units": ridge.coef_ / scaler.scale_})
