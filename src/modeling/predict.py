from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.config import METADATA_PATH, MODEL_PATH
from src.data.features import build_model_matrix


def model_artifacts_exist(
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> bool:
    """Check if model and metadata artifacts exist on disk."""
    return Path(model_path).exists() and Path(metadata_path).exists()


def load_metadata(metadata_path: str | Path = METADATA_PATH) -> dict:
    """Load model metadata JSON."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_p_make(
    shots: pd.DataFrame,
    season: str,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> pd.DataFrame:
    """Add p_make column to shots using saved XGBoost model artifacts."""
    if shots.empty:
        out = shots.copy()
        out["p_make"] = []
        return out

    if not model_artifacts_exist(model_path=model_path, metadata_path=metadata_path):
        raise FileNotFoundError(
            "Model artifacts are missing. Train a model first at src/modeling/train.py."
        )

    metadata = load_metadata(metadata_path)
    trained_season = metadata.get("season")
    if trained_season != season:
        raise ValueError(
            f"Model season mismatch: selected season={season}, trained season={trained_season}."
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
