"""One-time script to establish monitoring baseline.

Usage:
    python monitoring/establish_baseline.py [--season 2024-25] [--season-type "Regular Season"]

This script:
1. Loads the most recently trained model.
2. Runs inference on the full historical shot dataset.
3. Computes baseline Brier score, ECE, and feature distributions.
4. Saves monitoring/baseline_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import DB_PATH, get_model_artifact_paths
from src.data.features import add_engineered_features, build_model_matrix
from src.modeling.predict import load_metadata, model_artifacts_exist
from monitoring.monitor import (
    ModelMonitor,
    _ensure_monitoring_tables,
    initialize_monitoring_tables,
    populate_predictions_table,
)

_BASELINE_PATH = Path(__file__).parent / "baseline_metrics.json"


def _pick_best_season(db_path: Path) -> tuple[str, str]:
    """Return the season/type with the most shots in the database."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT season, season_type, COUNT(*) AS n FROM shots "
        "GROUP BY season, season_type ORDER BY n DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return "2023-24", "Regular Season"
    return row[0], row[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Establish ShotIQ monitoring baseline.")
    parser.add_argument("--season", default=None, help="Season string (e.g. 2024-25)")
    parser.add_argument("--season-type", default="Regular Season")
    args = parser.parse_args()

    season = args.season
    season_type = args.season_type

    if season is None:
        season, season_type = _pick_best_season(DB_PATH)
        print(f"No season specified — using {season} {season_type} (largest dataset).")

    if not model_artifacts_exist(season, season_type):
        print(f"ERROR: No model artifacts found for {season} {season_type}.")
        print("Run: python -m src.modeling.train to train a model first.")
        sys.exit(1)

    # Load model and metadata
    artifact_paths = get_model_artifact_paths(season, season_type)
    metadata = load_metadata(season, season_type)
    booster = xgb.Booster()
    booster.load_model(str(artifact_paths["model_path"]))

    print(f"Loaded model: {artifact_paths['model_path']}")
    print(f"Features: {metadata.get('features', [])}")

    # Ensure monitoring tables exist
    initialize_monitoring_tables(DB_PATH)

    # Populate predictions table (idempotent)
    print("\nScoring all shots and populating predictions table…")
    n_inserted = populate_predictions_table(DB_PATH, booster, metadata)
    print(f"  → {n_inserted} new predictions inserted.")

    # Load all predictions for this season
    conn = sqlite3.connect(DB_PATH)
    preds_df = pd.read_sql_query(
        "SELECT predicted_make_prob, actual_make FROM predictions "
        "WHERE season = ? AND season_type = ?",
        conn,
        params=(season, season_type),
    )
    shots_df = pd.read_sql_query(
        "SELECT loc_x, loc_y, shot_distance, shot_zone_basic, shot_type "
        "FROM shots WHERE season = ? AND season_type = ?",
        conn,
        params=(season, season_type),
    )
    conn.close()

    if preds_df.empty:
        print("ERROR: No predictions found after scoring. Exiting.")
        sys.exit(1)

    # Compute baseline metrics
    monitor = ModelMonitor(model=booster, db_path=DB_PATH)
    metrics = monitor.compute_calibration_metrics(
        preds_df["predicted_make_prob"].values,
        preds_df["actual_make"].values,
    )

    # Compute baseline feature distributions (summary stats for PSI reference)
    feature_distributions: dict[str, dict] = {}
    for col in ["loc_x", "loc_y", "shot_distance"]:
        if col in shots_df.columns:
            vals = shots_df[col].dropna().values
            feature_distributions[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p10": float(np.percentile(vals, 10)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
            }

    for col in ["shot_zone_basic", "shot_type"]:
        if col in shots_df.columns:
            vc = shots_df[col].value_counts(normalize=True)
            feature_distributions[col] = vc.round(6).to_dict()

    baseline = {
        "season": season,
        "season_type": season_type,
        "n_shots": int(len(preds_df)),
        "brier_score": round(metrics["brier_score"], 6),
        "ece": round(metrics["ece"], 6),
        "log_loss": round(metrics["log_loss"], 6),
        "feature_distributions": feature_distributions,
    }

    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))

    print("\n" + "=" * 50)
    print("Baseline established.")
    print(f"  Season      : {season} {season_type}")
    print(f"  Shots scored: {baseline['n_shots']:,}")
    print(f"  Brier score : {baseline['brier_score']:.4f}")
    print(f"  ECE         : {baseline['ece']:.4f}")
    print(f"  Log loss    : {baseline['log_loss']:.4f}")
    print(f"\nSaved to: {_BASELINE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
