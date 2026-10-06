"""
Download archived ECMWF IFS HRES forecasts for RDU from the Open-Meteo Single Runs API.

One run per day, initialised at 12 UTC. Each run is the full forecast exactly as it was
issued, so it contains no information from after its initialisation time.

Leakage rules enforced here:
  - Only runs initialised at least 12 h before the 2026-09-17 00:00 local cutoff are fetched.
    The last run (2026-09-16 12 UTC = 8am EDT) is published ~4-6 h after initialisation,
    i.e. mid-afternoon Sep 16, well before the cutoff.
  - The Historical Forecast API is deliberately NOT used: it stitches together the first
    hours of every run, which is close to observed weather.

Usage:  python fetch_ifs_runs.py
Output: nwp_data/ifs_12z_rdu.csv  (one row per run x valid hour)
Re-running resumes: runs already saved in nwp_data/runs/ are skipped.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 35.8922, -78.7819                 # RDU ASOS station (from the GHCNh files)
FIRST_RUN = pd.Timestamp("2024-03-15 12:00", tz="UTC")   # IFS archive starts 2024-03-14
CUTOFF = pd.Timestamp("2026-09-17 00:00", tz="America/New_York").tz_convert("UTC")
LAST_RUN = pd.Timestamp("2026-09-16 12:00", tz="UTC")
VARIABLES = ["temperature_2m", "dew_point_2m", "cloud_cover"]

OUT_DIR = Path("nwp_data")
RUN_DIR = OUT_DIR / "runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)

assert LAST_RUN <= CUTOFF - pd.Timedelta(hours=12), "last run must be issued well before the cutoff"


def fetch_run(init, retries=4):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "run": init.strftime("%Y-%m-%dT%H:%M"),
        "models": "ecmwf_ifs",
        "hourly": ",".join(VARIABLES),
        "forecast_days": 11,          # IFS 12Z runs go out 10 days; ask for a little more
        "timezone": "GMT",
        "timeformat": "unixtime",
    }
    for attempt in range(retries):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:      # rate limited: back off and retry
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 400:      # run not available in the archive
            return {"error": r.text}
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


runs = pd.date_range(FIRST_RUN, LAST_RUN, freq="D")
failed = []
for i, init in enumerate(runs):
    path = RUN_DIR / f"{init:%Y%m%d%H}.json"
    if path.exists():
        continue
    data = fetch_run(init)
    if "error" in data:
        failed.append(str(init))
        continue
    path.write_text(json.dumps(data))
    if i % 50 == 0:
        print(f"{i + 1}/{len(runs)}  {init:%Y-%m-%d %HZ}")
    time.sleep(0.3)                   # stay well under Open-Meteo's free rate limits

# Combine every saved run into one long table
rows = []
for path in sorted(RUN_DIR.glob("*.json")):
    data = json.loads(path.read_text())
    init = pd.to_datetime(path.stem, format="%Y%m%d%H", utc=True)
    h = pd.DataFrame(data["hourly"])
    h["valid_utc"] = pd.to_datetime(h.pop("time"), unit="s", utc=True)
    h["init_utc"] = init
    rows.append(h)

nwp = pd.concat(rows, ignore_index=True)
nwp["lead_h"] = ((nwp["valid_utc"] - nwp["init_utc"]) / pd.Timedelta(hours=1)).astype(int)
nwp = nwp[(nwp["lead_h"] >= 0) & nwp["temperature_2m"].notna()]
nwp = nwp.rename(columns={"temperature_2m": "nwp_temp", "dew_point_2m": "nwp_dewpoint",
                          "cloud_cover": "nwp_cloud"})
nwp = nwp[["init_utc", "valid_utc", "lead_h", "nwp_temp", "nwp_dewpoint", "nwp_cloud"]]

assert nwp["init_utc"].max() <= CUTOFF - pd.Timedelta(hours=12)
nwp.to_csv(OUT_DIR / "ifs_12z_rdu.csv", index=False)

print(f"\nRuns saved: {nwp['init_utc'].nunique()} / {len(runs)}  (unavailable: {len(failed)})")
print("Max lead per run (h):", nwp.groupby("init_utc")["lead_h"].max().describe()[["min", "50%", "max"]].to_dict())
print("Saved", OUT_DIR / "ifs_12z_rdu.csv", f"({len(nwp):,} rows)")
