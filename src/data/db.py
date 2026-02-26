from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH

logger = logging.getLogger(__name__)

SHOT_COLUMNS = [
    "season",
    "season_type",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "game_id",
    "game_event_id",
    "game_date",
    "loc_x",
    "loc_y",
    "shot_made_flag",
    "shot_distance",
    "shot_type",
    "action_type",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_id INTEGER,
    team_name TEXT,
    game_id TEXT,
    game_event_id INTEGER,
    game_date TEXT,
    loc_x REAL NOT NULL,
    loc_y REAL NOT NULL,
    shot_made_flag INTEGER NOT NULL,
    shot_distance REAL,
    shot_type TEXT,
    action_type TEXT,
    shot_zone_basic TEXT,
    shot_zone_area TEXT,
    shot_zone_range TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, season_type, player_id, game_id, game_event_id, loc_x, loc_y, shot_made_flag)
);

CREATE INDEX IF NOT EXISTS idx_shots_season ON shots(season, season_type);
CREATE INDEX IF NOT EXISTS idx_shots_player ON shots(player_id, season, season_type);
CREATE INDEX IF NOT EXISTS idx_shots_game ON shots(game_id);
"""


def _db_path(db_path: str | Path = DB_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(_db_path(db_path))


def initialize_database(db_path: str | Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _clean_records(frame: pd.DataFrame) -> list[tuple]:
    records: list[tuple] = []
    for row in frame.itertuples(index=False, name=None):
        cleaned = tuple(None if pd.isna(value) else value for value in row)
        records.append(cleaned)
    return records


def upsert_shots(frame: pd.DataFrame, db_path: str | Path = DB_PATH) -> int:
    """Insert shots with INSERT OR IGNORE to avoid duplicates."""
    if frame.empty:
        return 0

    initialize_database(db_path)

    safe = frame.copy()
    for col in SHOT_COLUMNS:
        if col not in safe.columns:
            safe[col] = pd.NA
    safe = safe[SHOT_COLUMNS]

    records = _clean_records(safe)
    placeholders = ",".join(["?"] * len(SHOT_COLUMNS))
    sql = f"INSERT OR IGNORE INTO shots ({','.join(SHOT_COLUMNS)}) VALUES ({placeholders})"

    with get_connection(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        conn.commit()
        inserted = conn.total_changes - before

    logger.info("Inserted %s shots into SQLite cache", inserted)
    return inserted


def delete_player_shots(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    db_path: str | Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM shots
            WHERE player_id = ? AND season = ? AND season_type = ?
            """,
            (int(player_id), season, season_type),
        )
        conn.commit()
        return int(cursor.rowcount)


def read_player_shots(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT * FROM shots
            WHERE player_id = ? AND season = ? AND season_type = ?
            ORDER BY game_date, game_id, game_event_id
            """,
            conn,
            params=(int(player_id), season, season_type),
        )


def read_season_shots(
    season: str,
    season_type: str = "Regular Season",
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT * FROM shots
            WHERE season = ? AND season_type = ?
            ORDER BY player_id, game_date, game_id, game_event_id
            """,
            conn,
            params=(season, season_type),
        )


def player_shots_exist(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    db_path: str | Path = DB_PATH,
) -> bool:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM shots
            WHERE player_id = ? AND season = ? AND season_type = ?
            LIMIT 1
            """,
            (int(player_id), season, season_type),
        ).fetchone()
    return row is not None


def season_shots_count(
    season: str,
    season_type: str = "Regular Season",
    db_path: str | Path = DB_PATH,
) -> int:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM shots WHERE season = ? AND season_type = ?",
            (season, season_type),
        ).fetchone()
    return int(row[0] if row else 0)
