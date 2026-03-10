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

TABLEAU_PUBLIC_VIEW_URL = (
    "https://public.tableau.com/views/NBAShotIQDashboard/Dashboard1"
    "?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
)
TABLEAU_EMBED_URL = (
    "https://public.tableau.com/views/NBAShotIQDashboard/Dashboard1"
    "?:showVizHome=no&:tabs=no&:toolbar=yes&:language=en-US"
)
GITHUB_REPO_URL = "https://github.com/praveenpuviindran/nba_shotiq"


def _season_type_slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def get_model_artifact_paths(season: str, season_type: str) -> dict[str, Path]:
    """
    Return artifact paths scoped by season and season type.

    Example model path:
    models/xgb_model_2025-26_regular_season.json
    """
    safe_type = _season_type_slug(season_type)
    suffix = f"{season}_{safe_type}"
    return {
        "model_path": MODELS_DIR / f"xgb_model_{suffix}.json",
        "metadata_path": MODELS_DIR / f"metadata_{suffix}.json",
        "calibration_plot_path": MODELS_DIR / f"calibration_curve_{suffix}.png",
    }
