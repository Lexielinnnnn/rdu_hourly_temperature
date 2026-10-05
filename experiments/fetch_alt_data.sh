#!/usr/bin/env bash
# Download every alternative data set used by experiments/backtest.py.  Run from the repo root:
#   bash experiments/fetch_alt_data.sh
# NOTE: several sources deliver data past the Sep 17 2026 forecast cutoff (2026 station files,
# "current" index files).  That is fine here: experiments/backtest.py clips EVERY source at the
# cutoff before any feature is built, and experiments/test_no_leakage.py proves it.
set -euo pipefail
mkdir -p raw_data/stations raw_data/era5_wind raw_data/era5_land raw_data/sst raw_data/indices
OM="https://archive-api.open-meteo.com/v1/archive"
END=2026-09-16   # last full day before the forecast window

# 1) NOAA GHCNh hourly obs at corridor stations (same format as the RDU files)
get_station() {  # $1 label, $2 station id
  for y in 2021 2022 2023 2024 2025 2026; do
    curl -sf -o "raw_data/stations/$1_$y.psv" \
      "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/$y/psv/GHCNh_$2_$y.psv" &
  done
}
get_station dca  USW00013743   # Washington Reagan
get_station gso  USW00013723   # Greensboro (wedge / Piedmont)
get_station clt  USW00013881   # Charlotte
get_station roa  USW00013741   # Roanoke (west of Blue Ridge)
get_station orf  USW00013737   # Norfolk (coast)
get_station crw  USW00013866   # Charleston WV (upstream NW)
get_station ilm  USW00013748   # Wilmington NC (Atlantic moisture)
wait

# 2) ERA5 100 m winds + MSLP at mid-Atlantic points
wind() { curl -sf -o "raw_data/era5_wind/$1.json" "$OM?latitude=$2&longitude=$3&start_date=2021-01-01&end_date=$END&hourly=wind_speed_100m,wind_direction_100m,temperature_2m,pressure_msl&timezone=UTC"; }
wind chesbay 37.0 -76.0 & wind hatteras 35.2 -75.0 & wind shenandoah 38.0 -79.0 & wind charlotte 35.2 -80.8 & wind dc 38.85 -77.04 &
wait

# 3) ERA5-Land soil moisture + cloud cover (land-surface memory, diurnal range)
land() { curl -sf -o "raw_data/era5_land/$1.json" "$OM?latitude=$2&longitude=$3&start_date=2021-01-01&end_date=$END&hourly=soil_moisture_0_to_7cm,soil_moisture_28_to_100cm,cloud_cover&timezone=UTC"; }
land rdu 35.88 -78.79 & land roanoke 37.3 -79.9 & land wilmington 34.2 -77.9 &
wait

# 4) NOAA OISST daily sea-surface temperature (Gulf Stream, Gulf of Mexico, SC and mid-Atlantic bights)
sst() { curl -sf -o "raw_data/sst/$1.csv" "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?sst%5B(2021-01-01):1:($END)%5D%5B(0.0)%5D%5B($2)%5D%5B($3)%5D"; }
sst gulfstream 35.125 -75.125 & sst gulfmex 27.125 -90.125 & sst scbight 32.125 -79.125 & sst midatl 37.125 -74.125 &
wait

# 5) NOAA CPC daily teleconnection indices + monthly Nino3.4 (files run past the cutoff; clipped later)
B=https://ftp.cpc.ncep.noaa.gov/cwlinks
curl -sf -o raw_data/indices/nao.csv "$B/norm.daily.nao.cdas.z500.19500101_current.csv"
curl -sf -o raw_data/indices/pna.csv "$B/norm.daily.pna.cdas.z500.19500101_current.csv"
curl -sf -o raw_data/indices/ao.txt  "$B/norm.daily.ao.index.b500101.current.ascii"
curl -sf -o raw_data/indices/nino.txt "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
du -sh raw_data/*/
