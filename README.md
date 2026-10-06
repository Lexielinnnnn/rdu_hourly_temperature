# RDU hourly temperature forecast (AIPI 520, Project 1, Team 4)

**Task:** predict the hourly air temperature at Raleigh-Durham airport (RDU) for every hour from 12am Sep 17 to 11pm
Sep 30 2026 (336 predictions), using only data available before 12am Sep 17. Required: one linear regression plus at
least one other model.

**Team:** Burak Donbekci, Lexie Lin, Samuel Teshome. This `dev` branch combines the work from the `burak`, `lexie`
and `sam-teshome` branches.

## Results

Real Sep 17–30 2026 weather, 322 scored hours (the downloaded observations end Sep 30 at 9:51am). Every forecast
was saved before the real temperatures were downloaded, and nothing was re-tuned afterwards.

| Model | Type | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|
| **MOS: ECMWF forecast + ridge correction** | linear regression (ridge) | **2.59** | **3.44** | **0.64** |
| Raw ECMWF (no correction, normal after day 11) | reference | 2.68 | 3.53 | 0.62 |
| Random forest (local-midnight fix) | random forest | 3.00 | 3.64 | 0.59 |
| Station-only two-stage linear | linear regression (ridge) | 3.33 | 4.05 | 0.50 |
| Baseline: same hour, ±7 days, prior years | baseline | 3.48 | 4.22 | 0.45 |
| Baseline: climatology (normal for the date) | baseline | 3.57 | 4.30 | 0.43 |
| Ridge + alternative data | linear regression (ridge) | 3.69 | 4.56 | 0.36 |
| KNN analogs | k-nearest neighbors | 3.72 | 4.57 | 0.36 |
| Baseline: repeat last 24 h | baseline | 4.06 | 4.96 | 0.25 |

**Proposed submitted models** (team to confirm): MOS as the linear regression and the random forest as the other model.

Backtest of MOS (42 forecasts, Aug 15 – Oct 15 of 2024 and 2025, retrained before each season): MAE 2.24 °C vs
2.39 raw ECMWF, 2.91 station-only linear and 3.03 climatology. Full table: `presentation/data/final_metrics_all.csv`.

## Repository map

| Path | What |
|---|---|
| `raw_data/`, `clean_data.ipynb`, `cleaned_data/rdu_51_clean.csv` | NOAA GHCN-hourly RDU observations 2021–Sep 16 2026; cleaning keeps the :51 hourly reports |
| `random_forest.ipynb`, `rdu_random_forest_predictions.csv` | Random forest notebook (with the local-midnight fix) and its forecast |
| `linear_regression.ipynb`, `rdu_linear_predictions.csv` | Station-only two-stage linear model (climatology + ridge on recent anomalies) |
| `fetch_ifs_runs.py`, `nwp_data/ifs_12z_rdu.csv` | Downloads archived ECMWF IFS 12 UTC forecast runs for RDU (Open-Meteo Single Runs API) |
| `nwp_mos.ipynb`, `rdu_mos_predictions.csv` | MOS model: ECMWF forecast + ridge correction, backtest and final forecast |
| `final_model/` | Clean modules for the final models (`rf_model.py`, `linear_model.py`, `mos_model.py`), cross-validation, leakage tests, final scoring. See `final_model/README.md` |
| `experiments/`, `results/`, `figures/`, `predictions_*.csv` | Alternative-data pipeline (nearby stations, ERA5, sea surface temperature, climate indices), ridge and KNN models, their CV and scoring. See `READMESam.md` and `PLAN.md` |
| `presentation/` | Data tables and scripts for the slide figures. See `presentation/README.md` |

## How to run (from the repo root)

```bash
pip install -r requirements.txt
```

**MOS model (best model)**
```bash
python fetch_ifs_runs.py                # only if nwp_data/ifs_12z_rdu.csv is missing (~10 min, resumable)
jupyter nbconvert --to notebook --execute --inplace nwp_mos.ipynb    # or open it and Run All (~4 min)
```

**Final models, cross-validation and leakage tests** (details in `final_model/README.md`)
```bash
python -W ignore final_model/test_no_leakage.py     # must end with ALL LEAKAGE TESTS PASSED
python -W ignore final_model/cross_validation.py
python -W ignore final_model/train_final.py
bash experiments/fetch_final_truth.sh               # real Sep 17-30 observations, scoring only
python -W ignore final_model/score_final.py
```

**Alternative-data experiments** (details in `READMESam.md`)
```bash
bash experiments/fetch_alt_data.sh                  # ~650 MB into raw_data/ (gitignored)
python -W ignore experiments/test_no_leakage.py
python -W ignore experiments/model_selection.py
python -W ignore experiments/final_forecast.py
```

**Presentation figures**
```bash
python -W ignore presentation/make_data.py          # optional, ~4 min: rebuilds presentation/data/
python presentation/make_figures.py                 # seconds: presentation/figures/*.png
```

## No data leakage

- Observations are clipped at 12am Sep 17 2026 as soon as they are loaded.
- The MOS model uses only ECMWF runs that started at least 12 hours before each forecast was issued. For the final
  forecast that is the Sep 16 12 UTC run (8am EDT, published that afternoon). This is asserted when the data is
  downloaded, when it is loaded, in the backtest and on the final forecast.
- All features use only hours before the forecast origin. Training targets never reach the test period.
- `final_model/test_no_leakage.py` (random forest, station-only linear) and `experiments/test_no_leakage.py` (ridge, KNN)
  overwrite all data after a forecast origin with garbage and check that the forecast does not change.
- The real Sep 17–30 temperatures are downloaded only by `experiments/fetch_final_truth.sh` and read only by the scoring
  scripts and `presentation/make_data.py`, after all forecasts were saved.

Units: °C throughout.
