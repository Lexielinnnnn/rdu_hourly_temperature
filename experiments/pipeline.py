"""Data loading and feature construction with a hard forecast cutoff.

The forecast is issued at CUTOFF = 12am Sep 17 2026 (America/New_York).  Nothing observed at or after
CUTOFF may reach a feature or the training set.  Three guards enforce that:

1. `clip()` removes every row at or after CUTOFF from every source as soon as it is loaded
   (the 2026 station files and the "current" index files run past Sep 30).
2. Slowly-published sources (daily SST, daily indices, monthly Nino3.4) are shifted by their
   publication lag, so a value is only usable after it could really have been downloaded.
3. Every feature is a trailing window ending at origin - 1h (rolling means, diffs, shifts >= 365 d).
   experiments/test_no_leakage.py proves this by poisoning all data after a chosen time and checking
   that the features at that time do not change.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
H = 14 * 24
CUTOFF = pd.Timestamp("2026-09-17 00:00", tz=TZ).tz_convert("UTC")
GRID = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), CUTOFF, freq="h")  # last grid hour == CUTOFF itself
STATIONS = ["dca", "gso", "clt", "roa", "orf", "crw", "ilm"]
CAL = ["lead", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def clip(obj):
    """Drop everything at or after the forecast cutoff (the leakage guard)."""
    return obj[obj.index < CUTOFF]


def _poison(obj, poison_from):
    """Test hook: overwrite numeric values from `poison_from` onward with garbage, BEFORE clipping."""
    if poison_from is None:
        return obj
    obj = obj.copy()
    num = obj.select_dtypes("number").columns
    obj.loc[obj.index >= poison_from, num] = 1000.0
    return obj


def hourly(df, minute_target=51):
    """One obs per hour: the report closest to :51 (KDCA reports at :52; 2026 files are sub-hourly)."""
    df = df.dropna(subset=["temperature"]).copy()
    df["dist"] = (df["DATE"].dt.minute - minute_target).abs()
    df["hour"] = df["DATE"].dt.floor("h")
    return df.sort_values(["hour", "dist"]).drop_duplicates("hour").set_index("hour").drop(columns=["dist", "DATE"])


def load_sources(poison_from=None):
    """Return dict of hourly/daily sources on the UTC grid, clipped at CUTOFF."""
    P = lambda x: clip(_poison(x, poison_from))
    S = {}

    rdu = pd.read_csv("cleaned_data/rdu_51_clean.csv")
    rdu["DATE"] = pd.to_datetime(rdu["DATE"], utc=True)
    S["rdu"] = hourly(P(rdu.set_index("DATE", drop=False))).reindex(GRID).pipe(clip)

    cols = ["DATE", "temperature", "dew_point_temperature", "sea_level_pressure"]
    for st in STATIONS:
        d = pd.concat(pd.read_csv(f"raw_data/stations/{st}_{y}.psv", sep="|", usecols=cols, low_memory=False)
                      for y in range(2021, 2027))
        d["DATE"] = pd.to_datetime(d["DATE"], utc=True)
        for c in cols[1:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        S[st] = hourly(P(d.set_index("DATE", drop=False))).reindex(GRID).pipe(clip)

    for k in ["chesbay", "hatteras", "shenandoah", "charlotte", "dc"]:
        w = pd.DataFrame(json.load(open(f"raw_data/era5_wind/{k}.json"))["hourly"])
        w.index = pd.to_datetime(w.pop("time"), utc=True)
        w = clip(_poison(w, poison_from))
        spd, rad = w["wind_speed_100m"] / 3.6, np.deg2rad(w["wind_direction_100m"])
        S["wind_" + k] = pd.DataFrame({"u": -spd * np.sin(rad), "v": -spd * np.cos(rad),
                                       "mslp": w["pressure_msl"]}).reindex(GRID).pipe(clip)

    for k in ["rdu", "roanoke", "wilmington"]:
        w = pd.DataFrame(json.load(open(f"raw_data/era5_land/{k}.json"))["hourly"])
        w.index = pd.to_datetime(w.pop("time"), utc=True)
        S["land_" + k] = clip(_poison(w, poison_from)).reindex(GRID).pipe(clip)

    # daily sources: value for day d is only usable from d + 2 days (publication lag)
    sst = {}
    for k in ["gulfstream", "gulfmex", "scbight", "midatl"]:
        d = pd.read_csv(f"raw_data/sst/{k}.csv", skiprows=[1])
        d.index = pd.to_datetime(d["time"], utc=True).dt.normalize()
        sst[k] = _poison(d[["sst"]], poison_from)["sst"]
    S["sst"] = _lag_daily(pd.DataFrame(sst), 2)

    idx = {}
    for k in ["nao", "pna"]:
        d = pd.read_csv(f"raw_data/indices/{k}.csv")
        idx[k] = pd.Series(d.iloc[:, 3].values, index=pd.to_datetime(dict(year=d.iloc[:, 0], month=d.iloc[:, 1], day=d.iloc[:, 2]), utc=True))
    d = pd.read_fwf("raw_data/indices/ao.txt", colspecs=[(0, 4), (4, 7), (7, 10), (10, 20)], header=None)  # fixed width: -99 is glued to the day
    idx["ao"] = pd.Series(d[3].where(d[3] > -90).values, index=pd.to_datetime(dict(year=d[0], month=d[1], day=d[2]), utc=True))
    S["idx"] = _lag_daily(_poison(pd.DataFrame(idx), poison_from), 2)

    n = pd.read_csv("raw_data/indices/nino.txt", sep=r"\s+")
    n.index = pd.to_datetime(dict(year=n["YR"], month=n["MON"], day=1), utc=True) + pd.offsets.MonthBegin(2)  # month m known from start of m+2
    S["nino"] = clip(clip(_poison(pd.DataFrame({"nino34_anom": n["ANOM.3"]}), poison_from)).reindex(GRID, method="ffill"))
    return S


def _lag_daily(df, lag_days):
    df = df.copy()
    df.index = df.index + pd.Timedelta(days=lag_days)
    return clip(clip(df).reindex(GRID, method="ffill"))


def roll(s, n):
    return s.rolling(n, min_periods=n // 2).mean()


def build_features(S):
    """Origin-level feature table O, year-prior table Y, and the group->columns map."""
    R, t = S["rdu"], S["rdu"]["temperature"]
    O = pd.DataFrame(index=GRID)
    O["rdu_t_last"] = t.ffill(limit=3)
    O["rdu_t_24"], O["rdu_t_7d"] = roll(t, 24), roll(t, 168)
    O["rdu_td_24"] = roll(R["dew_point_temperature"], 24)
    O["rdu_rh_24"] = roll(R["relative_humidity"], 24)
    O["rdu_slp_last"] = R["sea_level_pressure"].ffill(limit=3)
    O["rdu_slp_d24"] = O["rdu_slp_last"] - O["rdu_slp_last"].shift(24)
    O["rdu_ws_24"] = roll(R["wind_speed"], 24)
    G = {"rdu": list(O.columns)}

    # corridor stations: how warm / what pressure relative to RDU (fronts, wedge, Atlantic air)
    for st in STATIONS:
        O[f"{st}_minus_rdu_t24"] = roll(S[st]["temperature"], 24) - O["rdu_t_24"]
    O["dca_td_24"] = roll(S["dca"]["dew_point_temperature"], 24)
    for st in ["dca", "roa", "orf", "crw"]:
        O[f"{st}_minus_rdu_slp"] = S[st]["sea_level_pressure"].ffill(limit=3) - O["rdu_slp_last"]
    G["corridor"] = [c for c in O.columns if c not in G["rdu"]]

    before = set(O.columns)
    for k in ["chesbay", "hatteras", "shenandoah", "charlotte", "dc"]:
        O[f"w_{k}_u24"], O[f"w_{k}_v24"] = roll(S["wind_" + k]["u"], 24), roll(S["wind_" + k]["v"], 24)
    O["w_grad_ns"] = roll(S["wind_dc"]["mslp"] - S["wind_hatteras"]["mslp"], 24)
    O["w_grad_coast"] = roll(S["wind_chesbay"]["mslp"] - S["wind_shenandoah"]["mslp"], 24)
    G["wind"] = [c for c in O.columns if c not in before]

    before = set(O.columns)
    for k in ["rdu", "roanoke", "wilmington"]:
        O[f"land_{k}_sm_shallow_24"] = roll(S["land_" + k]["soil_moisture_0_to_7cm"], 24)
    O["land_rdu_sm_deep_24"] = roll(S["land_rdu"]["soil_moisture_28_to_100cm"], 24)
    O["land_rdu_sm_shallow_7d"] = roll(S["land_rdu"]["soil_moisture_0_to_7cm"], 168)
    O["land_rdu_cloud_24"] = roll(S["land_rdu"]["cloud_cover"], 24)
    G["land"] = [c for c in O.columns if c not in before]

    before = set(O.columns)
    for k in S["sst"].columns:
        O[f"sst_{k}"] = S["sst"][k]
    O["sst_gulfstream_d7"] = S["sst"]["gulfstream"] - S["sst"]["gulfstream"].shift(24 * 7)
    G["sst"] = [c for c in O.columns if c not in before]

    before = set(O.columns)
    for k in S["idx"].columns:
        O[f"idx_{k}"] = S["idx"][k]
        O[f"idx_{k}_7d"] = roll(S["idx"][k], 24 * 7)
    O["idx_nino34"] = S["nino"]["nino34_anom"]
    G["indices"] = [c for c in O.columns if c not in before]

    # year-prior: RDU at the target hour 1 and 2 years earlier (+-3 day smoothing)
    ext = pd.date_range(GRID[0], CUTOFF + pd.Timedelta(hours=H), freq="h")  # forecast window needs year-prior rows too
    tf = t.reindex(ext).interpolate(limit=3, limit_area="inside")
    sm = lambda days: pd.concat([tf.shift(24 * (days + k)) for k in range(-3, 4)], axis=1).mean(axis=1)
    Y = pd.DataFrame({"yp1_t": tf.shift(24 * 365), "yp1_t_sm7": sm(365), "yp2_t_sm7": sm(730)}, index=ext)
    Y["yp_mean_sm7"] = Y[["yp1_t_sm7", "yp2_t_sm7"]].mean(axis=1)
    G["yearprior"] = list(Y.columns)
    return O, Y, G


def make_features(O, Y, t, origin):
    """Feature matrix for the 336 hours starting at `origin`; features use data < origin only."""
    hours = pd.date_range(origin, periods=H, freq="h")
    loc = hours.tz_convert(TZ)
    X = pd.DataFrame(np.repeat(O.loc[[origin - pd.Timedelta(hours=1)]].values, H, 0),
                     columns=O.columns, index=hours).join(Y.reindex(hours))
    X["lead"] = np.arange(H)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * loc.hour / 24), np.cos(2 * np.pi * loc.hour / 24)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    return X, t.reindex(hours)
