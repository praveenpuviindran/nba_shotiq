# NBA ShotIQ

NBA ShotIQ is an interactive NBA shooting analysis app that compares shot **difficulty** and shot **results**.

The core question is simple:

**Is a player making shots at the rate we would expect, based on where those shots are taken?**

## Summary

- Problem: Raw FG% alone does not tell you how hard those shots were.
- Approach: Use shot-level data + an expected-make model to estimate shot quality by location.
- Outcome: View where a player shoots most, where shots are easiest/hardest, and where they overperform or underperform expectation.

## What This Project Does

- Loads player shot data for a chosen season and game type (Regular Season or Playoffs)
- Builds location-based shot maps in two styles (Hexbin and Court Zones)
- Computes key metrics: Attempts, FG%, xFG%, SMOE, Shot Diet Difficulty
- Ranks players in a sortable season leaderboard
- Highlights best and worst scoring areas for a selected player

## Live App

- Streamlit App: https://nbashotiq.streamlit.app/

## Tableau Dashboard

- Tableau Public: https://public.tableau.com/views/NBAShotIQDashboard/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
- Streamlit integration:
  - `Tableau Dashboard` page embeds the live Tableau view in-app
  - `Data Exports` page lets you generate Tableau CSV extracts from cached SQLite data

### Dashboard Screenshot (Placeholder)

Add a screenshot or GIF here once captured:

```text
![NBA ShotIQ Tableau Dashboard](reports/tableau/assets/dashboard_placeholder.png)
```

## Data

- Source: `nba_api` (`ShotChartDetail`, `LeagueGameLog`, player metadata)
- Scope: Single-season analysis per run (season options generated dynamically up to current season)
- Storage: Local SQLite cache (`data/nba_shotiq.db`) for faster reloads
- API Safety: Throttling + retry/backoff to reduce rate-limit failures

## Methods

### 1) Shot Feature Engineering

- Base coordinates: `LOC_X`, `LOC_Y`
- Derived features include distance, angle, and 3-point inference
- Optional categorical zone features are one-hot encoded

### 2) Expected Make Model

- Baseline: Logistic Regression
- Final model: XGBoost classifier
- Metrics: Log Loss and Brier score
- Diagnostics: Calibration curve image per season/game-type model

### 3) Shot Performance Views

- Shot Volume: where attempts are concentrated
- Expected Make Chance: model-estimated make probability by location
- Over/Under Expected: actual results minus expected results (SMOE)

## Metric Cheat Sheet

- `FG%`: Actual make rate
- `xFG%`: Expected make rate from the model
- `SMOE`: `FG% - xFG%` (positive means above expected)
- `Shot Diet Difficulty`: `1 - xFG%` (higher means tougher average shot profile)

## How To Run

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open `http://localhost:8501`.

### Export Data for Tableau

```bash
python scripts/export_tableau.py --season 2023 --out data/tableau_exports
```

Or via Makefile:

```bash
make export-tableau SEASON=2023 SEASON_TYPE="Regular Season"
```

### Train a Model Manually

```bash
python -m src.modeling.train --season 2025-26 --season-type "Regular Season"
```

Model artifacts are saved with season/type-specific names:

- `xgb_model_<season>_<season_type>.json`
- `metadata_<season>_<season_type>.json`
- `calibration_curve_<season>_<season_type>.png`

## Project Structure

```text
nba_shotiq/
  Makefile
  app/
    app.py
  scripts/
    export_tableau.py
  reports/
    tableau/
      README.md
  src/
    config.py
    nba/
      players.py
      shots_api.py
      seasons.py
    data/
      db.py
      ingest.py
      features.py
    modeling/
      train.py
      evaluate.py
      predict.py
    viz/
      court.py
      heatmaps.py
  data/
    .gitkeep
  models/
    .gitkeep
  tests/
    test_features.py
    test_db.py
    test_seasons.py
  requirements.txt
  README.md
  .gitignore
```

## Model Monitoring

ShotIQ includes production-style model monitoring to detect performance
degradation and feature drift over time.

**Establish baseline (run once after training):**
```bash
python monitoring/establish_baseline.py
```

**Run a monitoring check at any time:**
```bash
# Via Streamlit — "Model Health" tab → "Run Monitoring Check"
# Or programmatically:
python -c "
from monitoring.monitor import ModelMonitor
from src.config import DB_PATH
monitor = ModelMonitor(model=None, db_path=DB_PATH, window_days=30)
report = monitor.run_monitoring_report()
print(report)
"
```

**Metrics tracked:**
- **Brier Score** — mean squared error of predicted probabilities (calibration quality)
- **ECE** — Expected Calibration Error (binned, 10 deciles)
- **Log Loss** — penalizes confident wrong predictions
- **PSI per feature** — Population Stability Index for `loc_x`, `loc_y`, `shot_distance`,
  `shot_zone_basic`, `shot_type`

**Alert thresholds:**
- Brier score > baseline × 1.15 → alert
- Any feature PSI > 0.25 → alert

Reports are appended to `monitoring/monitoring_history.json` and the
`model_monitoring_log` SQLite table.

---

## SQL Analytics

All analytics queries live in the `sql/` directory and run directly against
the local SQLite database.

**Run queries from the command line:**
```bash
sqlite3 data/nba_shotiq.db < sql/01_player_shot_quality_trends.sql
sqlite3 data/nba_shotiq.db < sql/02_shot_context_cohort_analysis.sql
sqlite3 data/nba_shotiq.db < sql/03_player_overperformance_leaderboard.sql
sqlite3 data/nba_shotiq.db < sql/04_game_flow_shot_selection.sql
```

**Or use the Streamlit SQL Explorer tab.**

**Queries included:**

| # | Query | Technique |
|---|---|---|
| 01 | Player shot quality trends | LAG, rolling AVG window functions |
| 02 | Shot context cohort analysis | CASE WHEN cohorts, RANK OVER PARTITION BY |
| 03 | Overperformance leaderboard | CTEs, manual STDDEV, 95% CI filter, NTILE |
| 04 | Shot selection by season/zone | Conditional aggregation, 2-season rolling AVG |

> Note: Queries 01–03 require the `predictions` table. Run `establish_baseline.py` first.

---

## Causal Inference Analysis

Beyond correlation: ShotIQ includes a causal analysis estimating whether
improving shot selection quality *causally* improves scoring efficiency.

**Methods used:**
- **Inverse Probability Weighting (IPW)** — propensity score logistic model, bootstrap CI
- **Difference-in-Differences (DiD)** — player panel, consecutive seasons
- **Rosenbaum sensitivity bounds** — measures robustness against unmeasured confounders

**Key finding:** See the Causal Analysis tab in the Streamlit app, or run:
```python
from causal.shot_selection_analysis import *
from src.config import DB_PATH
panel = load_player_season_data(DB_PATH)
df_ps = compute_propensity_scores(panel)
print(estimate_ipw_effect(df_ps))
```

**Full methodology:** [`causal/CAUSAL_FINDINGS.md`](causal/CAUSAL_FINDINGS.md)

---

## Portfolio-Ready Bullet

- Built an end-to-end NBA shot quality analytics product with Streamlit + Tableau Public, including expected-value modeling (XGBoost), interactive leaderboards, and production-style export workflows for reproducible BI reporting.

## Notes for Readers

- First load can take longer because data may need to be downloaded and cached.
- Expected-make estimates are based on shot-level context (mainly location and zone features), not full tracking context (defender distance, time pressure, etc.).
- Treat outputs as decision-support signals, not a final judgment of player value.

## Troubleshooting (macOS)

If `xgboost` fails with a `libomp.dylib` error:

```bash
brew install libomp
```

Then restart your shell/venv and run Streamlit again.
