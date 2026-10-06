# Presentation figures

One small script per figure. Run from the repo root:

```bash
python presentation/make_figures.py              # all figures, a few seconds
python presentation/fig3_lead_day_skill.py       # or just one
```

The scripts read the small tables in `presentation/data/` (committed). To rebuild those tables from the models
(about 4 minutes; reruns the MOS backtest and checks that `final_model/mos_model.py` reproduces the frozen
`rdu_mos_predictions.csv` exactly):

```bash
python -W ignore presentation/make_data.py
```

Colors follow the model on every slide: MOS blue, raw ECMWF orange, station-only linear aqua, climatology and baselines
grey, actual temperatures black. Shared style in `style.py` (same validated palette as `final_model/plot_style.py`).

| Figure | Shows | Suggested slide |
|---|---|---|
| `fig1_forecast_vs_actual.png` | The frozen MOS forecast vs the real Sep 17–30 temperatures, with the heat wave and cold spell marked | Performance (hero) |
| `fig2_model_ladder.png` | Every model's error on the real window, ranked and grouped by input type | Performance |
| `fig3_lead_day_skill.png` | Backtest error by forecast day: MOS vs raw ECMWF vs station-only linear vs climatology | Evaluation approach |
| `fig4_trust_by_lead_day.png` | The MOS weight on ECMWF's forecast for each forecast day (0.97 on day 1, 0.38 on day 11) | Models |
| `fig5_ecmwf_bias_drift.png` | ECMWF's monthly error at RDU and the Aug 2025 shift behind the recent-bias feature | Models / lessons learned |
| `fig6_leakage_timeline.png` | What the forecast knew and when, around the Sep 17 cutoff | Data pipeline / leakage |
| `fig7_normal_vs_actual.png` | Temperature = normal for the date + a departure (Aug–Sep 2026) | Problem / features |
| `fig8_error_heatmap.png` | Each model's error on each day of the real window | Performance / limitations |
| `fig9_r2_skill.png` | R² on the real window, and how much each model adds beyond the normal for the date (R² is 0.43 for the normal alone) | Performance / Evaluation approach |

Figures are 200 dpi PNGs sized for a 16:9 slide.
