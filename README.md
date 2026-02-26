# NBA ShotIQ

NBA ShotIQ helps you answer a simple question:

**Is a player making shots at the rate we would expect, given where those shots are taken?**

It combines NBA shot chart data with a probability model and presents results in an interactive Streamlit app.

## What You Can Do

- Pick a player, season, and game type (Regular Season or Playoffs)
- See where the player shoots most often
- See expected make chance by court location
- See where the player is shooting better or worse than expected
- Compare players in a season leaderboard by Attempts, FG%, xFG%, SMOE, and Shot Diet Difficulty

## Metric Cheat Sheet

- `FG%`: Actual make rate
- `xFG%`: Expected make rate from the model
- `SMOE`: `FG% - xFG%` (positive = above expected)
- `Shot Diet Difficulty`: `1 - xFG%` (higher = tougher average shot profile)

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open `http://localhost:8501`.

## Data and Caching

- Data source: `nba_api` (`ShotChartDetail`)
- Player lookup: `nba_api.stats.static.players`
- Local cache: `data/nba_shotiq.db` (SQLite)
- API protection: throttling + retry/backoff to reduce rate-limit failures

Why caching matters: once shots are cached locally, reloads are much faster and require fewer API calls.

## Model Training

Train one model per season + game type:

```bash
python -m src.modeling.train --season 2025-26 --season-type "Regular Season"
```

Artifacts are saved in `models/` with season/type-specific names:

- `xgb_model_<season>_<season_type>.json`
- `metadata_<season>_<season_type>.json`
- `calibration_curve_<season>_<season_type>.png`

## Tests

```bash
pytest -q
```

## Known Constraints

- NBA Stats endpoints can throttle or temporarily block requests, especially for first-time full-season pulls.
- The expected-make model only uses shot-level context (location/zone and related features). It does not include defender distance, time remaining, or player tracking.
- Results are best interpreted directionally (patterns and tendencies), not as a perfect measure of player skill in isolation.

## Troubleshooting (macOS)

If `xgboost` fails to load with a `libomp.dylib` error:

```bash
brew install libomp
```

Then restart your shell/venv and run Streamlit again.
