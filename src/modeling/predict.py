from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import get_model_artifact_paths
from src.data.features import build_model_matrix


def _load_xgboost_module():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific dynamic import
        raise RuntimeError(
            "XGBoost could not be loaded. On macOS, install OpenMP with "
            "`brew install libomp`, then restart your shell/venv and retry."
        ) from exc
    return xgb


def model_artifacts_exist(
    season: str,
    season_type: str = "Regular Season",
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> bool:
    """Check if model and metadata artifacts exist on disk."""
    if model_path is None or metadata_path is None:
        artifacts = get_model_artifact_paths(season=season, season_type=season_type)
        model_path = artifacts["model_path"]
        metadata_path = artifacts["metadata_path"]
    return Path(model_path).exists() and Path(metadata_path).exists()


def load_metadata(
    season: str,
    season_type: str = "Regular Season",
    metadata_path: str | Path | None = None,
) -> dict:
    """Load model metadata JSON."""
    if metadata_path is None:
        metadata_path = get_model_artifact_paths(
            season=season,
            season_type=season_type,
        )["metadata_path"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_p_make(
    shots: pd.DataFrame,
    season: str,
    season_type: str = "Regular Season",
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add p_make column to shots using saved XGBoost model artifacts."""
    xgb = _load_xgboost_module()
    if model_path is None or metadata_path is None:
        artifacts = get_model_artifact_paths(season=season, season_type=season_type)
        model_path = artifacts["model_path"]
        metadata_path = artifacts["metadata_path"]

    if shots.empty:
        out = shots.copy()
        out["p_make"] = []
        return out

    if not model_artifacts_exist(
        season=season,
        season_type=season_type,
        model_path=model_path,
        metadata_path=metadata_path,
    ):
        raise FileNotFoundError(
            "Model artifacts are missing. Train a model first at src/modeling/train.py."
        )

    metadata = load_metadata(
        season=season,
        season_type=season_type,
        metadata_path=metadata_path,
    )
    trained_season = metadata.get("season")
    trained_season_type = metadata.get("season_type")
    if trained_season != season or trained_season_type != season_type:
        raise ValueError(
            "Model context mismatch: "
            f"selected=({season}, {season_type}), "
            f"trained=({trained_season}, {trained_season_type})."
        )

    feature_columns = metadata["features"]

    X, _, _ = build_model_matrix(shots, feature_columns=feature_columns)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmatrix = xgb.DMatrix(X, feature_names=feature_columns)
    probabilities = booster.predict(dmatrix)

    out = shots.copy()
    out["p_make"] = probabilities
    return out
