from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CALIBRATION_PLOT_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.data.features import build_model_matrix
from src.data.ingest import ensure_season_shots_cached
from src.modeling.evaluate import compute_binary_metrics, save_calibration_plot

logger = logging.getLogger(__name__)


def _load_xgboost_module():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific dynamic import
        raise RuntimeError(
            "XGBoost could not be loaded. On macOS, install OpenMP with "
            "`brew install libomp`, then restart your shell/venv and retry."
        ) from exc
    return xgb


def _split_train_test(
    shots: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by game_id when possible, otherwise by player_id."""
    if "game_id" in shots.columns and shots["game_id"].notna().any():
        groups = shots["game_id"].fillna(
            shots["player_id"].astype(str) + "_missing_game"
        )
    else:
        groups = shots["player_id"].fillna(-1).astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(shots, groups=groups))
    return shots.iloc[train_idx].reset_index(drop=True), shots.iloc[test_idx].reset_index(drop=True)


def train_model_for_season(
    season: str,
    season_type: str = "Regular Season",
    force_reingest: bool = False,
    max_players: int | None = None,
) -> dict:
    """Train baseline + XGBoost shot make models and save season artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    xgb = _load_xgboost_module()

    logger.info("Ensuring season shot cache exists for %s (%s)", season, season_type)
    shots = ensure_season_shots_cached(
        season=season,
        season_type=season_type,
        force_refresh=force_reingest,
        max_players=max_players,
    )

    if shots.empty:
        raise ValueError("No shots are available for training.")

    train_df, test_df = _split_train_test(shots, test_size=0.2)

    X_train, y_train, feature_columns = build_model_matrix(train_df)
    X_test, y_test, _ = build_model_matrix(test_df, feature_columns=feature_columns)

    if y_train is None or y_test is None:
        raise ValueError("Target column 'shot_made_flag' is missing.")

    logger.info("Training baseline Logistic Regression on %s rows", len(X_train))
    baseline = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_SEED)
    baseline.fit(X_train, y_train)
    baseline_probs = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_binary_metrics(y_test.to_numpy(), baseline_probs)

    logger.info("Training final XGBoost model on %s rows", len(X_train))
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=4,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_binary_metrics(y_test.to_numpy(), xgb_probs)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    xgb_model.get_booster().save_model(str(MODEL_PATH))

    save_calibration_plot(
        y_true=y_test.to_numpy(),
        y_prob=xgb_probs,
        output_path=CALIBRATION_PLOT_PATH,
    )

    metadata = {
        "season": season,
        "season_type": season_type,
        "features": feature_columns,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_shots_total": int(len(shots)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": {
            "baseline_logistic": baseline_metrics,
            "xgboost": xgb_metrics,
        },
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "metadata_path": str(METADATA_PATH),
            "calibration_plot_path": str(CALIBRATION_PLOT_PATH),
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Training complete. XGBoost metrics: log_loss=%.4f, brier=%.4f",
        xgb_metrics["log_loss"],
        xgb_metrics["brier_score"],
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA ShotIQ model for a single season")
    parser.add_argument("--season", required=True, help="Season string like 2025-26")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Season type, MVP currently expects 'Regular Season'",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Re-fetch and refresh cached season shots before training",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Optional cap for quicker local experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train_model_for_season(
        season=args.season,
        season_type=args.season_type,
        force_reingest=args.force_reingest,
        max_players=args.max_players,
    )
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
