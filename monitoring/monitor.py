"""Model monitoring module for NBA ShotIQ.

Tracks calibration quality (Brier score, ECE, log loss) and feature
distribution drift (PSI) across rolling time windows.  Appends structured
reports to monitoring/monitoring_history.json and the model_monitoring_log
SQLite table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MONITORING_DIR = Path(__file__).parent
_HISTORY_PATH = _MONITORING_DIR / "monitoring_history.json"

# PSI thresholds (industry standard)
_PSI_OK = 0.10
_PSI_WARN = 0.25

# Brier score degradation threshold (15 % above baseline triggers alert)
_BRIER_DEGRADATION_FACTOR = 1.15


@dataclass
class FeatureDriftResult:
    psi: float
    status: str  # "ok", "warn", "alert"


@dataclass
class MonitoringReport:
    timestamp: str
    season: str
    season_type: str
    brier_score: float
    ece: float
    log_loss: float
    n_predictions: int
    drift_report: dict[str, dict]  # feature_name -> {"psi": float, "status": str}
    alert_triggered: bool
    alert_message: str
    baseline_brier: Optional[float] = None


class ModelMonitor:
    """Production-style monitor for the ShotIQ XGBoost expected-make model."""

    def __init__(self, model, db_path: str | Path, window_days: int = 30):
        """
        Parameters
        ----------
        model:
            Loaded XGBoost model (or None — metrics are computed from DB predictions).
        db_path:
            Path to the SQLite database containing shots and predictions.
        window_days:
            Rolling window length for metric computation.
        """
        self.model = model
        self.db_path = Path(db_path)
        self.window_days = window_days

        baseline_path = _MONITORING_DIR / "baseline_metrics.json"
        self.baseline: Optional[dict] = None
        if baseline_path.exists():
            with open(baseline_path) as f:
                self.baseline = json.load(f)

    # ------------------------------------------------------------------ #
    # Calibration metrics                                                  #
    # ------------------------------------------------------------------ #

    def compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """
        Compute Brier score, Expected Calibration Error (ECE), and log loss.

        Parameters
        ----------
        predictions : array-like of float in [0, 1]
            Predicted make probabilities.
        actuals : array-like of int {0, 1}
            Observed shot outcomes.
        n_bins : int
            Number of equal-width bins for ECE computation (default: 10 = deciles).

        Returns
        -------
        dict with keys "brier_score", "ece", "log_loss".
        """
        p = np.asarray(predictions, dtype=float).clip(1e-7, 1 - 1e-7)
        y = np.asarray(actuals, dtype=float)

        # Brier score
        brier = float(np.mean((p - y) ** 2))

        # ECE — binned calibration error
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_sum = 0.0
        n = len(y)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            bin_mean_pred = p[mask].mean()
            bin_mean_actual = y[mask].mean()
            ece_sum += (mask.sum() / n) * abs(bin_mean_pred - bin_mean_actual)

        # Log loss
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        return {"brier_score": brier, "ece": ece_sum, "log_loss": ll}

    # ------------------------------------------------------------------ #
    # Feature drift (PSI)                                                  #
    # ------------------------------------------------------------------ #

    def detect_feature_drift(
        self,
        recent_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        n_bins: int = 10,
    ) -> dict[str, dict]:
        """
        Compute Population Stability Index (PSI) for each numeric feature.

        PSI = Σ (actual% − expected%) × ln(actual% / expected%)

        Interpretation:
            PSI < 0.10  → no significant change ("ok")
            0.10–0.25   → moderate change ("warn")
            > 0.25      → significant change ("alert")

        Parameters
        ----------
        recent_df : DataFrame
            Recent data slice (e.g. last window_days).
        baseline_df : DataFrame
            Historical reference distribution.
        n_bins : int
            Number of buckets for continuous features.

        Returns
        -------
        dict mapping feature_name → {"psi": float, "status": str}
        """
        numeric_cols = [
            c for c in ["loc_x", "loc_y", "shot_distance"]
            if c in recent_df.columns and c in baseline_df.columns
        ]

        results: dict[str, dict] = {}
        for col in numeric_cols:
            base_vals = baseline_df[col].dropna().values
            recent_vals = recent_df[col].dropna().values

            if len(base_vals) == 0 or len(recent_vals) == 0:
                results[col] = {"psi": 0.0, "status": "ok"}
                continue

            # Use baseline quantiles as bin edges
            bin_edges = np.quantile(base_vals, np.linspace(0, 1, n_bins + 1))
            bin_edges[0] -= 1e-6
            bin_edges[-1] += 1e-6

            expected_counts, _ = np.histogram(base_vals, bins=bin_edges)
            actual_counts, _ = np.histogram(recent_vals, bins=bin_edges)

            # Convert to percentages, clip to avoid log(0)
            expected_pct = (expected_counts / expected_counts.sum()).clip(1e-6)
            actual_pct = (actual_counts / actual_counts.sum()).clip(1e-6)

            psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

            if psi < _PSI_OK:
                status = "ok"
            elif psi < _PSI_WARN:
                status = "warn"
            else:
                status = "alert"

            results[col] = {"psi": round(psi, 6), "status": status}

        # Categorical features — treat each unique value as a bucket
        cat_cols = [
            c for c in ["shot_zone_basic", "shot_type"]
            if c in recent_df.columns and c in baseline_df.columns
        ]
        for col in cat_cols:
            base_vc = baseline_df[col].value_counts(normalize=True)
            recent_vc = recent_df[col].value_counts(normalize=True)
            all_cats = set(base_vc.index) | set(recent_vc.index)

            psi = 0.0
            for cat in all_cats:
                exp = max(base_vc.get(cat, 0.0), 1e-6)
                act = max(recent_vc.get(cat, 0.0), 1e-6)
                psi += (act - exp) * np.log(act / exp)

            psi = float(psi)
            status = "ok" if psi < _PSI_OK else ("warn" if psi < _PSI_WARN else "alert")
            results[col] = {"psi": round(psi, 6), "status": status}

        return results

    # ------------------------------------------------------------------ #
    # Full monitoring report                                               #
    # ------------------------------------------------------------------ #

    def run_monitoring_report(
        self,
        season: Optional[str] = None,
        season_type: str = "Regular Season",
    ) -> MonitoringReport:
        """
        Pull recent predictions from the database, compute all metrics,
        persist the report to JSON and SQLite, and return a MonitoringReport.
        """
        conn = sqlite3.connect(self.db_path)

        # Determine the date cutoff for the rolling window
        cutoff_sql = (
            f"date(game_date, '-{self.window_days} days')"
            if self.window_days
            else "'1900-01-01'"
        )

        season_filter = ""
        params: list = []
        if season:
            season_filter = "AND s.season = ? AND s.season_type = ?"
            params.extend([season, season_type])

        try:
            query = f"""
                SELECT
                    p.shot_id,
                    p.predicted_make_prob,
                    p.actual_make,
                    p.game_date,
                    s.loc_x,
                    s.loc_y,
                    s.shot_distance,
                    s.shot_zone_basic,
                    s.shot_type,
                    s.season,
                    s.season_type
                FROM predictions p
                JOIN shots s ON s.id = p.shot_id
                WHERE p.game_date >= {cutoff_sql}
                  {season_filter}
                ORDER BY p.game_date DESC
            """
            recent = pd.read_sql_query(query, conn, params=params if params else None)
        except Exception as exc:
            logger.warning("Could not read predictions table: %s", exc)
            conn.close()
            # Return a minimal report with defaults
            return MonitoringReport(
                timestamp=datetime.utcnow().isoformat(),
                season=season or "all",
                season_type=season_type,
                brier_score=float("nan"),
                ece=float("nan"),
                log_loss=float("nan"),
                n_predictions=0,
                drift_report={},
                alert_triggered=False,
                alert_message="No predictions table found. Run populate_predictions_table() first.",
            )

        if recent.empty:
            conn.close()
            return MonitoringReport(
                timestamp=datetime.utcnow().isoformat(),
                season=season or "all",
                season_type=season_type,
                brier_score=float("nan"),
                ece=float("nan"),
                log_loss=float("nan"),
                n_predictions=0,
                drift_report={},
                alert_triggered=False,
                alert_message=f"No predictions found in the last {self.window_days} days.",
            )

        # Calibration metrics on recent predictions
        metrics = self.compute_calibration_metrics(
            recent["predicted_make_prob"].values,
            recent["actual_make"].values,
        )

        # Baseline shots for drift comparison
        try:
            baseline_shots = pd.read_sql_query(
                f"SELECT loc_x, loc_y, shot_distance, shot_zone_basic, shot_type FROM shots"
                f"{(' WHERE season = ? AND season_type = ?' if season else '')}",
                conn,
                params=([season, season_type] if season else None),
            )
        except Exception:
            baseline_shots = pd.DataFrame()

        conn.close()

        drift_report = self.detect_feature_drift(recent, baseline_shots) if not baseline_shots.empty else {}

        # Alert logic
        alert_triggered = False
        alert_parts: list[str] = []

        baseline_brier = self.baseline.get("brier_score") if self.baseline else None
        if baseline_brier and not np.isnan(metrics["brier_score"]):
            if metrics["brier_score"] > baseline_brier * _BRIER_DEGRADATION_FACTOR:
                alert_triggered = True
                alert_parts.append(
                    f"Brier score {metrics['brier_score']:.4f} exceeds "
                    f"baseline {baseline_brier:.4f} × {_BRIER_DEGRADATION_FACTOR} "
                    f"({baseline_brier * _BRIER_DEGRADATION_FACTOR:.4f})."
                )

        for feat, info in drift_report.items():
            if info["status"] == "alert":
                alert_triggered = True
                alert_parts.append(f"Feature '{feat}' PSI = {info['psi']:.4f} (> {_PSI_WARN}).")

        alert_message = " ".join(alert_parts) if alert_parts else "All checks passed."

        report = MonitoringReport(
            timestamp=datetime.utcnow().isoformat(),
            season=season or "all",
            season_type=season_type,
            brier_score=round(metrics["brier_score"], 6),
            ece=round(metrics["ece"], 6),
            log_loss=round(metrics["log_loss"], 6),
            n_predictions=len(recent),
            drift_report=drift_report,
            alert_triggered=alert_triggered,
            alert_message=alert_message,
            baseline_brier=baseline_brier,
        )

        self._persist_report(report)
        return report

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def _persist_report(self, report: MonitoringReport) -> None:
        """Append report to monitoring_history.json and upsert into SQLite."""
        # JSON history
        history: list[dict] = []
        if _HISTORY_PATH.exists():
            try:
                history = json.loads(_HISTORY_PATH.read_text())
            except json.JSONDecodeError:
                history = []

        history.append(asdict(report))
        _HISTORY_PATH.write_text(json.dumps(history, indent=2))

        # SQLite log
        try:
            conn = sqlite3.connect(self.db_path)
            _ensure_monitoring_tables(conn)
            conn.execute(
                """
                INSERT INTO model_monitoring_log
                    (run_date, brier_score, ece, log_loss, alert_triggered,
                     alert_message, feature_drift_json, n_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.timestamp,
                    report.brier_score if not np.isnan(report.brier_score) else None,
                    report.ece if not np.isnan(report.ece) else None,
                    report.log_loss if not np.isnan(report.log_loss) else None,
                    int(report.alert_triggered),
                    report.alert_message,
                    json.dumps(report.drift_report),
                    report.n_predictions,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Could not write to model_monitoring_log: %s", exc)


# ------------------------------------------------------------------ #
# Database helpers                                                     #
# ------------------------------------------------------------------ #

_MONITORING_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_monitoring_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date           TEXT NOT NULL,
    brier_score        REAL,
    ece                REAL,
    log_loss           REAL,
    alert_triggered    INTEGER DEFAULT 0,
    alert_message      TEXT,
    feature_drift_json TEXT,
    n_predictions      INTEGER,
    created_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id             INTEGER NOT NULL,
    season              TEXT NOT NULL,
    season_type         TEXT NOT NULL,
    player_id           INTEGER NOT NULL,
    game_date           TEXT,
    predicted_make_prob REAL NOT NULL,
    actual_make         INTEGER NOT NULL,
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(shot_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_season  ON predictions(season, season_type);
CREATE INDEX IF NOT EXISTS idx_pred_player  ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_pred_date    ON predictions(game_date);
"""


def _ensure_monitoring_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_MONITORING_SCHEMA)
    conn.commit()


def initialize_monitoring_tables(db_path: str | Path) -> None:
    """Public helper — idempotently create monitoring tables."""
    with sqlite3.connect(Path(db_path)) as conn:
        _ensure_monitoring_tables(conn)


def populate_predictions_table(
    db_path: str | Path,
    model,
    metadata: dict,
    batch_size: int = 5000,
) -> int:
    """
    For every shot in the database, run the model and store
    (shot_id, predicted_make_prob, actual_make) in the predictions table.

    Already-scored shots (UNIQUE shot_id constraint) are silently skipped.

    Parameters
    ----------
    db_path : path to SQLite database
    model : loaded XGBoost model
    metadata : model metadata dict (contains 'features' list)
    batch_size : rows per batch to avoid memory pressure

    Returns
    -------
    int — number of new rows inserted
    """
    from src.data.features import add_engineered_features, build_model_matrix

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    _ensure_monitoring_tables(conn)

    total_shots = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    inserted_total = 0

    for offset in range(0, total_shots, batch_size):
        rows = pd.read_sql_query(
            f"SELECT id, season, season_type, player_id, game_date, "
            f"loc_x, loc_y, shot_made_flag, shot_distance, shot_type, "
            f"action_type, shot_zone_basic, shot_zone_area, shot_zone_range "
            f"FROM shots LIMIT {batch_size} OFFSET {offset}",
            conn,
        )
        if rows.empty:
            break

        rows = add_engineered_features(rows)
        X, _, feature_cols = build_model_matrix(rows, feature_columns=metadata.get("features"))

        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        probs = model.predict(dmatrix)

        records = [
            (
                int(rows.iloc[i]["id"]),
                str(rows.iloc[i]["season"]),
                str(rows.iloc[i]["season_type"]),
                int(rows.iloc[i]["player_id"]),
                rows.iloc[i].get("game_date", None),
                float(probs[i]),
                int(rows.iloc[i]["shot_made_flag"]),
            )
            for i in range(len(rows))
        ]

        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO predictions
                (shot_id, season, season_type, player_id, game_date,
                 predicted_make_prob, actual_make)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
        inserted_total += conn.total_changes - before
        logger.info("Scored shots %d–%d (%d inserted)", offset, offset + len(rows), inserted_total)

    conn.close()
    return inserted_total
