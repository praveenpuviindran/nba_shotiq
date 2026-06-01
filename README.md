# NBA ShotIQ

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/live%20app-streamlit-red)](https://nbashotiq.streamlit.app/)
[![Tableau Dashboard](https://img.shields.io/badge/dashboard-Tableau%20Public-blue)](https://public.tableau.com/views/NBAShotIQDashboard/Dashboard1)

Live Streamlit app: https://nbashotiq.streamlit.app/

---

## Project Summary

NBA ShotIQ is an end-to-end NBA shot quality analytics platform that separates **shot difficulty** from **shot results**.

Raw field-goal percentage is a noisy metric — it conflates a player's shooting skill with the quality of the shots they take. ShotIQ computes an **expected field goal probability (xFG%)** for every shot attempt using spatial and contextual features, then surfaces where each player over- or under-performs that expectation.

**Core question:** Is a player making shots at the rate we would expect, given where those shots are taken?

### What This Project Does

- Ingests player shot data from the NBA stats API for any season and game type
- Builds spatial shot maps in two views: hexbin density and court-zone partitions
- Trains and evaluates a calibrated expected-make model per season/game-type combination
- Computes player leaderboard metrics: FG%, xFG%, SMOE (Shooting Make Over Expected), and Shot Diet Difficulty
- Embeds the Tableau Public dashboard for BI-style reporting
- Exports clean CSV extracts for Tableau
- Tracks model health over time with a production-style monitoring module

---

## Methodology

### 1) Shot Feature Engineering

Each shot attempt is represented by:

- **Spatial coordinates** — `LOC_X`, `LOC_Y` (NBA half-court coordinate system, rim at origin)
- **Derived features** — shot distance (Euclidean from rim), shot angle, 3-point inference flag
- **Categorical zone features** — `shot_zone_basic`, `shot_zone_area`, `shot_zone_range` (one-hot encoded)

### 2) Expected-Make Model (xFG%)

The model pipeline trains two classifiers on historical shot-level data:

| Model | Role |
|---|---|
| Logistic Regression | Baseline calibration reference |
| XGBoost (`binary:logistic`) | Final production model |

**Training protocol:**
- Shots are split by `game_id` (GroupShuffleSplit) to prevent data leakage across games.
- XGBoost hyperparameters: n_estimators=350, learning_rate=0.05, max_depth=6, subsample=0.9.
- A separate model is trained per season × game-type combination for temporal accuracy.

**Evaluation metrics:**

| Metric | Description |
|---|---|
| **Log Loss** | Penalises confident wrong predictions; lower is better |
| **Brier Score** | Mean squared error of predicted probabilities; lower is better |
| **Calibration Curve** | Plots predicted probability against observed make frequency across bins; ideally follows the diagonal |

Calibration curves are saved automatically as PNG artifacts during training:
`models/calibration_curve_<season>_<season_type>.png`

### 3) Shot Performance Metrics

| Metric | Definition |
|---|---|
| `FG%` | Actual make rate |
| `xFG%` | Model-estimated expected make rate |
| `SMOE` | `FG% − xFG%` — positive means shooting above expectation |
| `Shot Diet Difficulty` | `1 − xFG%` — higher means harder average shot profile |

### 4) Model Monitoring

A production-style monitoring module tracks model degradation and feature drift:

- **Brier Score drift** — alert if current > baseline × 1.15
- **ECE (Expected Calibration Error)** — binned across 10 deciles
- **PSI per feature** — Population Stability Index for `loc_x`, `loc_y`, `shot_distance`, `shot_zone_basic`, `shot_type`; alert if PSI > 0.25

---

## Tableau Dashboard

The Streamlit app embeds the live Tableau dashboard for BI-style shot analysis. A CSV export pipeline generates Tableau-ready extracts from the local SQLite cache.

**Live dashboard:** https://public.tableau.com/views/NBAShotIQDashboard/Dashboard1

### Dashboard Screenshot

> Add a screenshot or GIF once captured:

```text
![NBA ShotIQ Tableau Dashboard](reports/tableau/assets/dashboard_screenshot.png)
```

---

## Setup

### Prerequisites

- Python 3.10+
- (macOS) `brew install libomp` if XGBoost raises a `libomp.dylib` error

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501`. The first load may take longer as shot data is fetched and cached.

### Train a model manually

```bash
python -m src.modeling.train --season 2025-26 --season-type "Regular Season"
```

Artifacts saved to:
- `models/xgb_model_<season>_<season_type>.json`
- `models/metadata_<season>_<season_type>.json`
- `models/calibration_curve_<season>_<season_type>.png`

### Export data for Tableau

```bash
python scripts/export_tableau.py --season 2023 --out data/tableau_exports
# or via Makefile:
make export-tableau SEASON=2023 SEASON_TYPE="Regular Season"
```

### Run tests

```bash
pytest tests/
```

---

## Causal Inference Analysis

Beyond correlation, ShotIQ includes a causal analysis estimating whether improving shot selection quality *causally* improves scoring efficiency.

**Methods used:**
- **Inverse Probability Weighting (IPW)** — propensity-score logistic model with bootstrap confidence intervals
- **Difference-in-Differences (DiD)** — player panel across consecutive seasons
- **Rosenbaum sensitivity bounds** — measures robustness against unmeasured confounders

See [`causal/CAUSAL_FINDINGS.md`](causal/CAUSAL_FINDINGS.md) for full methodology.

---

## SQL Analytics

All analytics queries run directly against the local SQLite database:

```bash
sqlite3 data/nba_shotiq.db < sql/01_player_shot_quality_trends.sql
```

| Query | Technique |
|---|---|
| Player shot quality trends | LAG, rolling AVG window functions |
| Shot context cohort analysis | CASE WHEN cohorts, RANK OVER PARTITION BY |
| Overperformance leaderboard | CTEs, manual STDDEV, 95% CI filter, NTILE |
| Shot selection by season/zone | Conditional aggregation, 2-season rolling AVG |

---

## Project Structure

```text
nba_shotiq/
  app/app.py                # Streamlit UI entry point
  src/
    config.py
    nba/
      shots_api.py          # NBA API data pipeline (throttled, retry/backoff)
      players.py
      seasons.py
    data/
      db.py                 # SQLite cache layer
      ingest.py
      features.py
    modeling/
      train.py              # Baseline + XGBoost training pipeline
      evaluate.py           # Log loss, Brier score, calibration curve
      predict.py
    viz/
      court.py              # NBA court geometry utilities
      heatmaps.py           # Hexbin and zone-partition shot maps
  causal/
    shot_selection_analysis.py
  monitoring/
    establish_baseline.py
    monitor.py
  scripts/export_tableau.py
  sql/
  tests/
  requirements.txt
  Makefile
```

---

## Limitations

- Expected-make estimates are based on shot location and zone features — not full tracking context (defender distance, time pressure, shot clock).
- First load requires fetching all player shots for the selected season via the NBA stats API; subsequent loads use the local SQLite cache.
- Treat outputs as decision-support signals, not a final judgment of player shooting quality.
