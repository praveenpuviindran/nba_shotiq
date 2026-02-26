from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "nba_shotiq.db"
MODEL_PATH = MODELS_DIR / "xgb_model.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
CALIBRATION_PLOT_PATH = MODELS_DIR / "calibration_curve.png"

DEFAULT_SEASON = "2025-26"
DEFAULT_SEASON_TYPE = "Regular Season"

NBA_API_SLEEP_SECONDS = float(os.getenv("NBA_API_SLEEP_SECONDS", "0.70"))
NBA_API_MAX_RETRIES = int(os.getenv("NBA_API_MAX_RETRIES", "5"))
NBA_API_BACKOFF_BASE_SECONDS = float(os.getenv("NBA_API_BACKOFF_BASE_SECONDS", "1.00"))

RANDOM_SEED = 42
