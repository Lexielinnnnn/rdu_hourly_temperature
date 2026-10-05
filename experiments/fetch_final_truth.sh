#!/usr/bin/env bash
# Real RDU observations (NOAA GHCNh, USW00013722, 2026) for FINAL SCORING ONLY.
# Kept apart from fetch_alt_data.sh on purpose: only experiments/score_final.py reads this file.
# pipeline.py never loads it, and every pipeline source is clipped at the Sep 17 2026 cutoff anyway.
set -euo pipefail
mkdir -p raw_data/final_truth
curl -sf -o raw_data/final_truth/rdu_2026.psv \
  "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2026/psv/GHCNh_USW00013722_2026.psv"
ls -la raw_data/final_truth
