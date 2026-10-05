#!/usr/bin/env bash
# Download the two alternative data sets used by experiments/backtest.py.
#   1) NOAA GHCNh hourly obs for Washington Reagan National (KDCA, USW00013743)
#   2) ERA5 reanalysis (via Open-Meteo archive API) 100 m winds + MSLP at 5 mid-Atlantic points
# Run from the repo root:  bash experiments/fetch_alt_data.sh
set -euo pipefail
mkdir -p raw_data/dca raw_data/era5_wind

for y in 2021 2022 2023 2024 2025 2026; do
  curl -sf -o "raw_data/dca/$y.psv" \
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/$y/psv/GHCNh_USW00013743_$y.psv" &
done
wait

fetch_point() {
  curl -sf -o "raw_data/era5_wind/$1.json" \
    "https://archive-api.open-meteo.com/v1/archive?latitude=$2&longitude=$3&start_date=2021-01-01&end_date=2026-09-16&hourly=wind_speed_100m,wind_direction_100m,temperature_2m,pressure_msl&timezone=UTC"
}
fetch_point chesbay    37.0  -76.0 &   # mouth of Chesapeake Bay (coastal / backdoor fronts)
fetch_point hatteras   35.2  -75.0 &   # offshore Cape Hatteras (Atlantic / tropical inflow)
fetch_point shenandoah 38.0  -79.0 &   # west of the Blue Ridge (NW cold-front approach)
fetch_point charlotte  35.2  -80.8 &   # upstream SW (Gulf moisture / warm advection)
fetch_point dc         38.85 -77.04 &  # Washington DC (N-S pressure gradient anchor)
wait
ls -la raw_data/dca raw_data/era5_wind
