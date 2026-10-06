"""RDU hourly data for both final models, with a hard cutoff at 12am Sep 17 2026 (Raleigh time).

Both final models (the random forest, the two-stage linear model) use ONLY RDU's own observations
from cleaned_data/rdu_51_clean.csv.  Everything at or after CUTOFF is dropped as soon as the file is read.
"""
import numpy as np
import pandas as pd

TZ = "America/New_York"
HORIZON = 14 * 24                                           # 336 hourly forecasts
CUTOFF = pd.Timestamp("2026-09-17 00:00", tz=TZ).tz_convert("UTC")
CSV = "cleaned_data/rdu_51_clean.csv"
WEATHER = ["temperature", "dew_point_temperature", "relative_humidity", "sea_level_pressure", "wind_speed"]
CLIM_COLS = ["temperature", "dew_point_temperature", "sea_level_pressure"]


def load(poison_from=None):
    """One row per observation, indexed by the UTC hour it belongs to.

    poison_from: test hook.  Overwrites all weather values at/after this time with garbage BEFORE the cutoff
    clip, so the leakage test can prove that nothing after a forecast origin affects that forecast.
    """
    d = pd.read_csv(CSV)
    d["DATE"] = pd.to_datetime(d["DATE"], utc=True)
    d = d.sort_values("DATE")
    d[WEATHER] = d[WEATHER].apply(pd.to_numeric, errors="coerce")
    if poison_from is not None:
        d.loc[d["DATE"] >= poison_from, WEATHER] = 1000.0
    d = d.loc[d["DATE"] < CUTOFF].copy()                    # the leakage guard
    d.index = d["DATE"].dt.floor("h")
    d.index.name = "hour"
    d = d[~d.index.duplicated(keep="last")]
    assert d.index.max() < CUTOFF
    return d


def hourly_grid(data):
    """Complete UTC hourly grid (missing hours stay NaN) so shifts and rolling windows are in real hours."""
    grid = pd.date_range(data.index.min(), data.index.max(), freq="h")
    return data.reindex(grid)[CLIM_COLS]


def local_midnights(first_utc, last_utc):
    """Daily forecast origins issued at LOCAL (Raleigh) midnight, as UTC timestamps, within [first, last]."""
    days = pd.date_range(first_utc.tz_convert(TZ).date(), last_utc.tz_convert(TZ).date(), freq="D", tz=TZ).tz_convert("UTC")
    return days[(days >= first_utc) & (days <= last_utc)]


def utc_midnights(first_utc, last_utc):
    """What the pushed random forest notebook did: origins at UTC midnight (8pm Raleigh in September)."""
    return pd.date_range(first_utc.ceil("D"), last_utc, freq="24h")


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[ok] - b[ok])))


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
