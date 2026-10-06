# Project 1 plan: RDU hourly temperature, Sep 17-30 2026

Branch: `sam-teshome`. Deliverables due **Wed Oct 7**: presentation, code repo, 2-4 page writeup.
Task: forecast 336 hourly RDU temperatures from 12am Sep 17 to 11pm Sep 30 using only data before Sep 17 12am.
Requirement: 1 linear regression + 1 other model.

## 1. Alternative data sets (all downloaded by `bash experiments/fetch_alt_data.sh`, about 650 MB, gitignored)

| Group | Data | Source | Why it should matter |
|---|---|---|---|
| corridor | Hourly obs at DC, Greensboro, Charlotte, Roanoke, Norfolk, Charleston WV, Wilmington NC | NOAA GHCNh (same format as RDU) | Upstream stations show fronts and the Appalachian wedge (cool air dammed east of the mountains) before RDU. |
| wind | 100 m wind and sea-level pressure at Chesapeake Bay, Hatteras, Shenandoah, Charlotte, DC | ERA5 via Open-Meteo | Wind direction tells which air mass is arriving; N-S and coast-inland pressure gradients. |
| land | Soil moisture (shallow and deep) at RDU, Roanoke, Wilmington; RDU cloud cover | ERA5-Land via Open-Meteo | Land surface is the main source of predictability from 5 to 14 days; dry soil means bigger day-night swings. NC was in drought this September. |
| sst | Daily sea-surface temperature: Gulf Stream, Gulf of Mexico, SC bight, mid-Atlantic bight | NOAA OISST | Warm water feeds humid southeasterly air and tropical systems. |
| indices | NAO, AO, PNA (daily) and Nino 3.4 anomaly (monthly) | NOAA CPC | Large-scale patterns are the only known source of skill at 10-30 days. |
| yearprior | RDU at the target hour 1 and 2 years earlier (+-3 day smoothing) | RDU data already in repo | The "look at the year prior" idea. |

Not done: tropical cyclone tracks (NHC best-track has no 2025-26 file yet), MJO (BoM file blocked), upper-air heights.

## 2. Data leakage protection (cutoff = 12am Sep 17 2026)

`experiments/pipeline.py` enforces it and `experiments/test_no_leakage.py` proves it:
1. Every source is clipped to strictly before the cutoff as soon as it is loaded (2026 station files and CPC files run past Sep 30).
2. Daily and monthly sources are shifted by a 2-day or 2-month publication lag, so a value is only used after it could really have been downloaded.
3. All features are trailing windows ending 1 hour before the forecast origin.
4. Poison test: all data from a chosen time onward is overwritten with garbage, and the features and training rows for that origin must not change. It passes for the real forecast and for backtest origins in 2023 and 2025. The test also checks that the poison does change later features, so it cannot pass by accident.
5. The backtest asserts that no training target reaches the test origin.

The real Sep 17-30 2026 RDU observations are downloaded separately (`experiments/fetch_final_truth.sh`) and are used only by `score_final.py` for final scoring, never for features or tuning. (`raw_data/2026.csv` ends at the cutoff, it does not contain them.)

## 3. Features and model choice

- **Features**: RDU state (last, 24 h and 7 d mean temperature, dew point, humidity, pressure change, wind), year-prior, plus each group above, plus hours ahead and sin/cos of hour of day and day of year.
- **Model 1: ridge linear regression** (required; scaled features).
- **Model 2: KNN regressor, k=300** on a small feature set. I read the 520 midterm material (Formula Reference NEIGHBORS, Week 5, quickref); nothing there beats KNN for an "analog forecast". The random forest already in the repo stays as an extra comparison.

### Backtest (`experiments/backtest.py`): mean absolute error in C, Sep 17-30 of 2022-2025

| Model and features | MAE | Day 1 | Day 2-3 | Day 4-14 |
|---|---|---|---|---|
| Baseline: same hour last 2 years | 3.12 | 3.35 | 2.63 | 3.19 |
| Baseline: repeat last 24 h | 3.93 | 2.28 | 3.01 | 4.25 |
| Ridge, RDU only | 3.17 | 2.18 | 2.21 | 3.44 |
| Ridge, + year-prior | 3.12 | 2.10 | 2.19 | 3.38 |
| **Ridge, + land (soil, cloud)** | **3.02** | **1.96** | 2.16 | 3.27 |
| **Ridge, + wind** | **3.03** | 2.25 | 2.13 | 3.26 |
| Ridge, + corridor stations | 3.23 | 2.14 | 2.30 | 3.50 |
| Ridge, + sst | 3.27 | 2.13 | 2.34 | 3.55 |
| Ridge, + indices | 3.40 | 2.12 | 2.51 | 3.68 |
| Ridge, all groups | 3.44 | 2.02 | 2.61 | 3.72 |
| KNN k=300, compact | 3.12 | 2.75 | 2.12 | 3.34 |
| **KNN, compact + DC temperature** | **3.03** | 2.74 | 2.09 | 3.23 |
| KNN, compact + indices | 3.25 | **1.97** | 2.16 | 3.57 |
| KNN, compact + sst / + soil | 3.18 / 3.41 | | | |
| Average of ridge + KNN | 3.05 | 2.23 | **1.98** | 3.32 |

### What this tells us (say this honestly in the writeup)
1. Best single changes are small: soil/cloud and wind help the linear model (3.12 to about 3.02) and DC temperature helps KNN (3.12 to 3.03). Year-to-year swings (2.0 to 4.4) are bigger than these gaps, and there are only 4 test seasons, so treat them as suggestive.
2. **Adding everything hurts** (all groups 3.44): many correlated features overfit with only about 1,900 training days. Add groups one at a time, keep few features, and use Lasso or ridge to prune.
3. Short range is where the data helps: soil/cloud cut day-1 error to 1.96, and the climate indices cut KNN day-1 error to 1.97, but indices hurt days 4-14. A model that uses different feature sets for short and long lead (or blends by lead) is the obvious next idea.
4. Days 4-14 stay near 3.2-3.4 C for every model, about the same as the year-prior baseline; that is close to the natural day-to-day noise.
5. The ridge + KNN average (3.05) is the most robust: it wins in 2022 and 2023 and has the best days 2-3.

### Data caveats
- `clean_data.ipynb` keeps only `:51` observations; KDCA reports at `:52` and 2026 files are sub-hourly, so exact-minute filtering drops hours. `hourly()` in `pipeline.py` takes the report nearest `:51`.
- Cleaned RDU data is in Celsius; confirm whether the grader wants Fahrenheit.

## 4. Split of remaining work (Mon Oct 5 / Tue Oct 6, then Wed Oct 7 due)

Owners are suggestions; adjust to taste.

### Sam (data + KNN + features) 
- **Mon**: turn `experiments/backtest.py` into clean pipeline modules (`src/data.py`, `src/features.py`); build the final KNN, tune k and the feature subset with time-ordered CV over several origins (not just Sep 17).
- **Tue**: feature-importance/ablation table (permutation importance or drop-group results above), anomaly-from-climatology experiment, final KNN predictions to `predictions_knn.csv`.

### Burak (linear regression + evaluation)
- **Mon**: final ridge/lasso/OLS comparison on feature sets A-D (use LassoCV to show which features survive, course Week 3 regularization), check collinearity (course Week 3), residual diagnostics.
- **Tue**: evaluation harness: score all models on actual Sep 17-30 2026 with MAE/RMSE/R2 and skill vs the two baselines (same-hour last year, repeat last 24 h), per-day error plot. Write the evaluation-approach section.
- Output: `predictions_linreg.csv`, `results_table.csv`.

### Lexie (existing RF/data cleaning + writeup + slides)
- **Mon**: fix `clean_data.ipynb` (nearest-:51 rule, fill gaps) and re-export the cleaned RDU file; rerun `random_forest.ipynb` on the new data as an extra model; start the writeup (problem, data pipeline, how course topics were applied: baselines, bias-variance, regularization, time-series CV, KNN scaling).
- **Tue**: build slides (inputs, pipeline, features, models, evaluation approach, performance, what did not work); draft the 2-4 page writeup; integrate the figures.

### Wed Oct 7 (all): final run from a clean clone, README with run order, proofread. Slides and writeup must be in our own words (AI is allowed only for code).

## 5. Reproduce
```bash
bash experiments/fetch_alt_data.sh
python -W ignore experiments/test_no_leakage.py   # must say ALL LEAKAGE TESTS PASSED
python -W ignore experiments/backtest.py          # writes experiments/results/backtest_results.csv
```
