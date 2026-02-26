# nba-shotiq

NBA ShotIQ is an MVP portfolio project for comparing **shot quality** (expected make probability) versus **shot making** (actual FG outcomes) for NBA players in a selected season.

## What It Does

- Fetches shot-level data from `nba_api` (`ShotChartDetail`) for a selected player/season.
- Caches shots locally in SQLite at `data/nba_shotiq.db` so repeated app runs do not repeatedly hit the API.
- Trains an expected field-goal model for one season at a time (baseline Logistic Regression + final XGBoost).
- Visualizes:
  - Shot Frequency Heatmap (hexbin density)
  - Shot Quality Heatmap (mean predicted `p_make` by location)
  - Shot Making Over Expected (SMOE) Heatmap (mean `made - p_make` by location)
- Displays key cards: Attempts, FG%, xFG%, SMOE.

## Repo Structure

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
  requirements.txt
  README.md
  .gitignore
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Data Fetching and Caching

- Player lookup uses `nba_api.stats.static.players`.
- Shot data is pulled from `nba_api.stats.endpoints.ShotChartDetail`.
- SQLite cache is created at runtime in `data/nba_shotiq.db` (gitignored).
- API calls include throttling, retries, and exponential backoff to reduce 429/403 failures.

## Train the Model

Train for one season (MVP expects `Regular Season`):

```bash
python -m src.modeling.train --season 2025-26 --season-type "Regular Season"
```

Artifacts saved to `models/`:
- `xgb_model.json`
- `metadata.json` (season, features, metrics, timestamp)
- `calibration_curve.png`

## Coordinate Mapping

`ShotChartDetail` returns `LOC_X`/`LOC_Y` already in NBA half-court coordinates (rim near `(0,0)`, positive Y toward midcourt). The app uses this direct mapping:

- `plot_x = LOC_X`
- `plot_y = LOC_Y`

See `src/viz/court.py` (`transform_shot_coords`) for the mapping function.

## Tests

```bash
pytest -q
```

Included tests cover:
- Feature engineering math (`distance`, `angle`, `is_3` inference)
- DB schema initialization and insert/read roundtrip behavior

## Limitations

- NBA Stats API can throttle/block requests (429/403), especially during large season ingests.
- Model uses shot-level spatial/zone features only; it does not include defender proximity, shot clock context, or tracking data.
- Single-season MVP only (architecture is modular to extend toward career mode later).
