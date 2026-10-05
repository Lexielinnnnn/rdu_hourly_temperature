# Project 1 plan: RDU hourly temperature, Sep 17-30 2026

Branch: `sam-teshome`. Deliverables due **Wed Oct 7**: presentation, code repo, 2-4 page writeup.
Task: forecast 336 hourly RDU temperatures from 12am Sep 17 to 11pm Sep 30 using only data before Sep 17 12am.
Requirement: 1 linear regression + 1 other model.

## 1. Two alternative data sets (beyond `raw_data/` RDU)

| # | Data set | Source | Why |
|---|----------|--------|-----|
| 1 | **Washington DC (KDCA) hourly obs**: temp, dew point, sea-level pressure | NOAA GHCNh, station USW00013743 (same format as the RDU files) | Upstream "hotspot": fronts usually reach DC before RDU. RDU-DC pressure/temperature differences capture frontal position. |
| 2 | **Mid-Atlantic winds + pressure**: 100 m wind (u/v) and MSLP at Chesapeake Bay mouth, offshore Cape Hatteras, west of the Blue Ridge (Shenandoah), Charlotte, DC | ERA5 reanalysis via Open-Meteo archive API (free, no key) | Wind direction = air-mass origin (NW continental dry vs SE Atlantic humid). Pressure gradients N-S (DC - Hatteras) and coast-inland (Bay - Shenandoah). |

`bash experiments/fetch_alt_data.sh` downloads both (about 80 MB, gitignored).
A third source, the **year-prior RDU series** (same hour 365 and 730 days earlier, +-3 day smoothing), needs no download and is built from the RDU data already in the repo.

## 2. Features (all are known at forecast origin Sep 16 11pm, so no leakage)

- **RDU state**: last-hour temp, 24 h mean temp, 7 d mean temp, 24 h mean dew point and RH, last pressure, 24 h pressure change, 24 h mean wind speed.
- **Year prior** (target-hour): `yp1_t_sm7`, `yp2_t_sm7`, their mean `yp_mean_sm7`. This is the "look at last year" idea.
- **DC**: 24 h mean temp and dew point, 24 h pressure change, DC-minus-RDU temp and pressure.
- **Wind and gradients**: 24 h mean u/v wind at 5 points, N-S and coast-inland pressure gradients.
- **Calendar and horizon**: hours ahead (`lead`), sin/cos of hour of day and day of year.

## 3. Model choice

**Model 1: linear regression** (ridge, alpha=1, scaled features). Required, and it was the strongest model.
**Model 2: KNN regressor.** I checked the 520 midterm material (Formula Reference NEIGHBORS, Week 5, quickref R3/R4). The course covers linear/ridge/lasso, KNN, SVM, logistic and time series (AR, seasonal naive). Nothing else there beats KNN here. KNN is the natural "analog forecast" (find the past situations most like today, average what happened next) and fits the geospatial idea. Random forest is already in the repo (`random_forest.ipynb`); keep it as an extra comparison, not as the required model.

### Backtest evidence (`experiments/backtest.py`)
Repeat the real task on Sep 17-30 of 2022-2025, training only on earlier origins. Mean absolute error in degrees C, average of the 4 years:

| Model / features | MAE | Day 1 | Day 2-3 | Day 4-14 |
|---|---|---|---|---|
| Baseline: same hour last 2 yrs | 3.12 | 3.35 | 2.63 | 3.19 |
| **Ridge LinReg, RDU + year-prior** | **3.12** | **2.10** | 2.19 | 3.38 |
| **KNN k=300, compact features** | **3.12-3.14** | 2.5-2.75 | **2.12** | 3.34 |
| Ridge, RDU only | 3.17 | 2.18 | 2.21 | 3.44 |
| Ridge, + DC + wind | 3.20 | 2.33 | 2.31 | 3.45 |
| Random forest, RDU only | 3.35 | 2.64 | 2.41 | 3.58 |
| KNN k=50, all features | 3.17 (RDU only) / 4.22 (+DC+wind) | | | |
| Repeat last 24 h | 3.93 | 2.28 | 3.01 | 4.25 |

### What this tells us (be honest in the writeup)
1. **Year-prior features help** the linear model (3.17 -> 3.12; day-1 error 2.18 -> 2.10) but only modestly. Test-year variance (2.0 to 4.4) is much larger than the gap between models, so with 4 test seasons these differences are within noise.
2. **DC and wind did not help overall.** They are informative at lead 0-3 days physically, but they are measured at one instant (origin) and a 14-day forecast from a single origin has no way to know the future weather, so most of the 14 days is basically climatology. Adding many correlated features hurt KNN badly (4.22) because distance gets diluted. This is a legitimate result for the slides: report it as "tested, did not help, and why".
3. **KNN needs few features and large k.** k=25 -> 3.34, k=100 -> 3.29, k=300 -> 3.14. Tune k with time-ordered CV (not random K-fold; the Week 4 time-series lesson).
4. Recommendation: submit **ridge LinReg (RDU + year-prior [+ DC if it survives tuning])** and **KNN (compact features, k tuned around 200-500)**. Optionally average the two; the errors are not identical across years (e.g. KNN wins 2025, LinReg wins 2024).
5. **Biggest remaining lever** (untested): model the diurnal cycle and a smoothed climatology separately, and predict the *anomaly*. Day 4-14 error is about 3.3 C for every model, which is roughly the day-to-day weather noise. Nobody has tried predicting the residual from climatology; I suggest this for Monday.

### Important data caveats found
- `clean_data.ipynb` keeps only :51 observations. KDCA reports at :52, and RDU 2026 has gaps; `hourly()` in `experiments/backtest.py` instead takes the report closest to :51 each hour. DC coverage for Jun-Sep 2026 is 30-90% missing under an exact-minute filter. Use the nearest-minute approach.
- `raw_data/2026.csv` already contains Sep 17-30 2026 observations. **Use those only for final scoring**, never for training or tuning, and say so in the writeup.
- The cleaned file is in Celsius. Check whether the grader wants Fahrenheit.

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
- **Tue**: build slides (inputs, pipeline, features, models, evaluation approach, performance, what did not work); draft the 2-4 page writeup; integrate Burak's and Sam's figures.

### Wed Oct 7 (all): final run from a clean clone, README with run order, proofread. Slides and writeup must be in our own words (AI is allowed only for code).

## 5. Reproduce
```bash
bash experiments/fetch_alt_data.sh
python -W ignore experiments/backtest.py   # about 4 minutes; writes experiments/results/backtest_results.csv
```
