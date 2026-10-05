"""Rolling-origin backtest: which features and which "other" model to use.

Forecast task (matches the project): at local midnight on Sep 17, predict the
next 336 hourly RDU temperatures using only data from before that moment.

Backtest: repeat that exact task at Sep 17 of 2022, 2023, 2024 and 2025,
training each time only on forecast origins whose 14-day window ends before
the test origin (expanding window, no leakage).

Run from the repo root (after `bash experiments/fetch_alt_data.sh`):
    python -W ignore experiments/backtest.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "America/New_York"
H = 14 * 24
TEST_YEARS = [2022, 2023, 2024, 2025]
grid = pd.date_range("2021-01-01", "2026-09-17 04:00", freq="h", tz="UTC")


def hourly(df, minute_target=51):
    """One observation per hour: the report closest to :51 (KDCA reports at :52,
    and much of 2026 is only 5-minute data, so an exact-minute filter loses hours)."""
    df = df.dropna(subset=["temperature"]).copy()
    df["dist"] = (df["DATE"].dt.minute - minute_target).abs()
    df["hour"] = df["DATE"].dt.floor("h")
    df = df.sort_values(["hour", "dist"]).drop_duplicates("hour")
    return df.set_index("hour").reindex(grid)


# ---------- existing RDU data ----------
rdu = pd.read_csv("cleaned_data/rdu_51_clean.csv")
rdu["DATE"] = pd.to_datetime(rdu["DATE"], utc=True)
R = hourly(rdu)

# ---------- alternative data set 1: Washington DC (KDCA) ----------
dca = pd.concat(
    pd.read_csv(f"raw_data/dca/{y}.psv", sep="|", low_memory=False,
                usecols=["DATE", "temperature", "dew_point_temperature", "sea_level_pressure"])
    for y in range(2021, 2027)
)
dca["DATE"] = pd.to_datetime(dca["DATE"], utc=True)
for c in ["temperature", "dew_point_temperature", "sea_level_pressure"]:
    dca[c] = pd.to_numeric(dca[c], errors="coerce")
D = hourly(dca)

# ---------- alternative data set 2: ERA5 mid-Atlantic winds ----------
W = {}
for k in ["chesbay", "hatteras", "shenandoah", "charlotte", "dc"]:
    w = pd.DataFrame(json.load(open(f"raw_data/era5_wind/{k}.json"))["hourly"])
    w.index = pd.to_datetime(w.pop("time"), utc=True)
    spd, rad = w["wind_speed_100m"] / 3.6, np.deg2rad(w["wind_direction_100m"])
    W[f"{k}_u"] = -spd * np.sin(rad)  # eastward component (m/s)
    W[f"{k}_v"] = -spd * np.cos(rad)  # northward component (m/s)
    W[f"{k}_mslp"] = w["pressure_msl"]
W = pd.DataFrame(W).reindex(grid)


# ---------- origin features: state of the atmosphere at origin - 1h ----------
def roll(s, n):
    return s.rolling(n, min_periods=n // 2).mean()


t = R["temperature"]
O = pd.DataFrame(index=grid)
O["rdu_t_last"] = t.ffill(limit=3)
O["rdu_t_24"] = roll(t, 24)
O["rdu_t_7d"] = roll(t, 168)
O["rdu_td_24"] = roll(R["dew_point_temperature"], 24)
O["rdu_rh_24"] = roll(R["relative_humidity"], 24)
O["rdu_slp_last"] = R["sea_level_pressure"].ffill(limit=3)
O["rdu_slp_d24"] = O["rdu_slp_last"] - O["rdu_slp_last"].shift(24)
O["rdu_ws_24"] = roll(R["wind_speed"], 24)
GROUPS = {"rdu": list(O.columns)}

O["dc_t_24"] = roll(D["temperature"], 24)
O["dc_td_24"] = roll(D["dew_point_temperature"], 24)
O["dc_slp_d24"] = D["sea_level_pressure"].ffill(limit=3).diff(24)
O["dc_minus_rdu_t24"] = O["dc_t_24"] - O["rdu_t_24"]
O["dc_minus_rdu_slp"] = D["sea_level_pressure"].ffill(limit=3) - O["rdu_slp_last"]
GROUPS["dc"] = [c for c in O.columns if c.startswith("dc")]

for c in W.columns:
    if c.endswith(("_u", "_v")):
        O[f"w_{c}_24"] = roll(W[c], 24)
O["w_grad_ns"] = roll(W["dc_mslp"] - W["hatteras_mslp"], 24)
O["w_grad_coast"] = roll(W["chesbay_mslp"] - W["shenandoah_mslp"], 24)
GROUPS["wind"] = [c for c in O.columns if c.startswith("w_")]

# ---------- year-prior features: RDU at the *target* hour 1-2 years earlier ----------
tf = t.interpolate(limit=3)


def same_hour_years_ago(days, half_window=3):
    return pd.concat([tf.shift(24 * (days + k)) for k in range(-half_window, half_window + 1)], axis=1).mean(axis=1)


Y = pd.DataFrame(index=grid)
Y["yp1_t"] = tf.shift(24 * 365)
Y["yp1_t_sm7"] = same_hour_years_ago(365)
Y["yp2_t_sm7"] = same_hour_years_ago(730)
Y["yp_mean_sm7"] = Y[["yp1_t_sm7", "yp2_t_sm7"]].mean(axis=1)
GROUPS["yearprior"] = list(Y.columns)
CAL = ["lead", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def features(origin):
    hours = pd.date_range(origin, periods=H, freq="h")
    loc = hours.tz_convert(TZ)
    X = pd.DataFrame(np.repeat(O.loc[[origin - pd.Timedelta(hours=1)]].values, H, 0),
                     columns=O.columns, index=hours).join(Y.reindex(hours))
    X["lead"] = np.arange(H)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * loc.hour / 24), np.cos(2 * np.pi * loc.hour / 24)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * loc.dayofyear / 365.25), np.cos(2 * np.pi * loc.dayofyear / 365.25)
    return X, t.reindex(hours)


origins = pd.date_range("2021-01-09", "2026-09-17", freq="D", tz=TZ).tz_convert("UTC")
cache = {o: features(o) for o in origins}


def build(os_):
    X, y = map(pd.concat, zip(*(cache[o] for o in os_)))
    keep = y.notna()
    return X[keep], y[keep]


SETS = {
    "A rdu only": GROUPS["rdu"] + CAL,
    "B + year-prior": GROUPS["rdu"] + GROUPS["yearprior"] + CAL,
    "C + DC": GROUPS["rdu"] + GROUPS["yearprior"] + GROUPS["dc"] + CAL,
    "D + DC + wind": GROUPS["rdu"] + GROUPS["yearprior"] + GROUPS["dc"] + GROUPS["wind"] + CAL,
}
KNN_COMPACT = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead",
               "rdu_t_24", "rdu_td_24", "rdu_slp_d24", "yp_mean_sm7", "dc_t_24"]


def linreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def knn(k):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(k, n_jobs=-1))


def rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(60, max_depth=16, min_samples_leaf=20, max_features=0.6,
                                               max_samples=0.3, n_jobs=-1, random_state=0))


EXPERIMENTS = [(s, "LinReg (ridge)", linreg, cols) for s, cols in SETS.items()]
EXPERIMENTS += [(s, "KNN k=50 (all feats)", lambda: knn(50), SETS[s]) for s in ["A rdu only", "D + DC + wind"]]
EXPERIMENTS += [("compact", f"KNN k={k}", (lambda k=k: knn(k)), KNN_COMPACT) for k in [25, 100, 300]]
EXPERIMENTS += [("compact no DC", "KNN k=300", lambda: knn(300), [c for c in KNN_COMPACT if c != "dc_t_24"])]
EXPERIMENTS += [(s, "Random forest", rf, SETS[s]) for s in ["A rdu only", "D + DC + wind"]]

rows = []
for yr in TEST_YEARS:
    test_origin = pd.Timestamp(f"{yr}-09-17", tz=TZ).tz_convert("UTC")
    X_tr, y_tr = build([o for o in origins if o <= test_origin - pd.Timedelta(hours=H)])
    X_te, y_te = cache[test_origin]
    keep = y_te.notna()
    X_te, y_te = X_te[keep], y_te[keep].values
    lead = X_te["lead"].values

    def record(feats, model, pred):
        err = np.abs(pred - y_te)
        rows.append(dict(year=yr, features=feats, model=model, MAE=err.mean(),
                         day1=err[lead < 24].mean(), day2_3=err[(lead >= 24) & (lead < 72)].mean(),
                         day4_14=err[lead >= 72].mean()))

    record("baseline", "same hour, last 2 yrs (+-3 d)", X_te["yp_mean_sm7"].values)
    last24 = t.reindex(pd.date_range(test_origin - pd.Timedelta(hours=24), periods=24, freq="h")).values
    record("baseline", "repeat last 24 h", np.tile(last24, 14)[keep.values])
    for feats, name, make, cols in EXPERIMENTS:
        record(feats, name, make().fit(X_tr[cols], y_tr).predict(X_te[cols]))
        print(yr, feats, name, round(rows[-1]["MAE"], 2), flush=True)

res = pd.DataFrame(rows)
Path("experiments/results").mkdir(exist_ok=True)
res.to_csv("experiments/results/backtest_results.csv", index=False)
summary = res.groupby(["features", "model"])[["MAE", "day1", "day2_3", "day4_14"]].mean()
by_year = res.pivot_table(index=["features", "model"], columns="year", values="MAE")
print("\nMean absolute error (deg C), averaged over Sep 17-30 of", TEST_YEARS)
print(summary.join(by_year).sort_values("MAE").round(2).to_string())
