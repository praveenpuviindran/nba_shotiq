"""SQL query runner for NBA ShotIQ analytics queries.

Usage example:
    from sql.runner import run_query
    from src.config import DB_PATH

    df = run_query("sql/01_player_shot_quality_trends.sql", DB_PATH)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def run_query(
    query_file: str | Path,
    db_path: str | Path,
    params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Execute a .sql file and return results as a DataFrame.

    Parameters
    ----------
    query_file : str or Path
        Path to the .sql file to execute.
    db_path : str or Path
        Path to the SQLite database.
    params : dict, optional
        Named parameters to substitute into the query (uses SQLite's
        ``?``-style positional binding — pass as a list for positional,
        or a dict for :name-style named params).

    Returns
    -------
    pd.DataFrame
        Query results. Empty DataFrame if the query returns no rows.
    """
    sql = Path(query_file).read_text()
    conn = sqlite3.connect(Path(db_path))
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def list_queries(sql_dir: str | Path = Path(__file__).parent) -> list[dict[str, str]]:
    """Return metadata for all .sql files in sql_dir.

    Extracts the 'Purpose:' line from each file's comment header so the
    Streamlit dropdown can show a human-readable label.

    Returns
    -------
    list of {"file": str, "label": str}
    """
    sql_dir = Path(sql_dir)
    queries = []
    for path in sorted(sql_dir.glob("*.sql")):
        label = path.stem  # fallback
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip().lstrip("-").strip()
            if stripped.lower().startswith("purpose:"):
                inline = stripped[len("purpose:"):].strip()
                if inline:
                    label = inline
                elif idx + 1 < len(lines):
                    label = lines[idx + 1].strip().lstrip("-").strip()
                break
        queries.append({"file": str(path), "label": label})
    return queries
