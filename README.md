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
  app/
    app.py
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
