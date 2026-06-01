from __future__ import annotations

import logging
import random
import time
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, ShotChartDetail

from src.config import (
    NBA_API_BACKOFF_BASE_SECONDS,
    NBA_API_MAX_RETRIES,
    NBA_API_SLEEP_SECONDS,
)

logger = logging.getLogger(__name__)


class NbaApiError(RuntimeError):
    """Raised when NBA API requests fail after retries."""


# Column rename map: NBA API returns UPPER_SNAKE names; we use lower_snake internally.
RAW_TO_STD_COLS = {
    "GAME_ID": "game_id",
    "GAME_EVENT_ID": "game_event_id",
    "GAME_DATE": "game_date",
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_NAME": "team_name",
    "LOC_X": "loc_x",
    "LOC_Y": "loc_y",
    "SHOT_MADE_FLAG": "shot_made_flag",
    "SHOT_DISTANCE": "shot_distance",
    "SHOT_TYPE": "shot_type",
    "ACTION_TYPE": "action_type",
    "SHOT_ZONE_BASIC": "shot_zone_basic",
    "SHOT_ZONE_AREA": "shot_zone_area",
    "SHOT_ZONE_RANGE": "shot_zone_range",
}

# Canonical output schema — every returned DataFrame has exactly these columns in this order.
BASE_SHOT_COLUMNS = [
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


def _throttle() -> None:
    """Sleep between API calls to respect the NBA stats API rate limit."""
    time.sleep(max(0.0, NBA_API_SLEEP_SECONDS))


def _run_with_retry(func: Callable[[], pd.DataFrame], op_name: str) -> pd.DataFrame:
    """Call func() up to NBA_API_MAX_RETRIES times with exponential backoff + jitter.

    Raises NbaApiError if all attempts fail (e.g. HTTP 429 throttle or 403 block).
    """
    last_error: Exception | None = None
    for attempt in range(1, NBA_API_MAX_RETRIES + 1):
        try:
            # Apply base throttle before every attempt to avoid rate-limit spikes.
            _throttle()
            return func()
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            last_error = exc
            if attempt == NBA_API_MAX_RETRIES:
                break
            # Exponential backoff: 2^(attempt-1) * base, plus random jitter to desync retries.
            sleep_for = NBA_API_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            sleep_for += random.uniform(0.0, 0.35)
            logger.warning(
                "NBA API request failed (%s), retry %s/%s in %.2fs",
                exc,
                attempt,
                NBA_API_MAX_RETRIES,
                sleep_for,
            )
            time.sleep(sleep_for)

    message = (
        f"NBA API request failed for '{op_name}' after {NBA_API_MAX_RETRIES} attempts. "
        "This can happen due to API throttling (HTTP 429) or blocking (HTTP 403)."
    )
    if last_error:
        message = f"{message} Last error: {last_error}"
    raise NbaApiError(message)


def _standardize_shot_frame(
    frame: pd.DataFrame,
    season: str,
    season_type: str,
    player_id: int,
    player_name: str,
) -> pd.DataFrame:
    """Rename columns, inject metadata fields, and enforce the canonical shot schema.

    Returns an empty DataFrame with the correct columns if the input is empty.
    """
    if frame.empty:
        return pd.DataFrame(columns=BASE_SHOT_COLUMNS)

    # Rename raw API column names to the project's internal lowercase schema.
    standardized = frame.rename(columns=RAW_TO_STD_COLS).copy()

    # Inject season context columns that are not returned by the ShotChartDetail endpoint.
    standardized["season"] = season
    standardized["season_type"] = season_type
    standardized["player_id"] = player_id
    standardized["player_name"] = player_name

    # Fill any columns present in the schema but absent from the raw frame with NA.
    for col in BASE_SHOT_COLUMNS:
        if col not in standardized.columns:
            standardized[col] = pd.NA

    # Enforce column order to match BASE_SHOT_COLUMNS exactly.
    standardized = standardized[BASE_SHOT_COLUMNS]

    # Coerce all spatial and numeric fields to float/int; malformed strings become NaN.
    numeric_cols = ["loc_x", "loc_y", "shot_made_flag", "shot_distance", "game_event_id", "team_id", "player_id"]
    for col in numeric_cols:
        standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    # shot_made_flag must be 0 or 1 — replace NaN with 0 (no-make assumption).
    standardized["shot_made_flag"] = standardized["shot_made_flag"].fillna(0).astype(int)
    return standardized


def fetch_player_shots(
    player_id: int,
    player_name: str,
    season: str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch shot-level data from ShotChartDetail for one player-season."""

    def _request() -> pd.DataFrame:
        # Request all field goal attempts (FGA) for this player using ShotChartDetail.
        endpoint = ShotChartDetail(
            team_id=0,
            player_id=int(player_id),
            context_measure_simple="FGA",
            season_nullable=season,
            # nba_api arg maps to API request param `SeasonTypeAllStar`.
            season_type_all_star=season_type,
        )
        frames = endpoint.get_data_frames()
        if not frames:
            return pd.DataFrame()
        # The first data frame contains the shot chart rows; remaining frames are secondary.
        return frames[0]

    raw = _run_with_retry(
        _request,
        op_name=f"ShotChartDetail(player_id={player_id}, season={season}, season_type={season_type})",
    )
    # Normalise raw API output into the canonical schema before returning.
    return _standardize_shot_frame(
        raw,
        season=season,
        season_type=season_type,
        player_id=player_id,
        player_name=player_name,
    )


def fetch_players_with_games(
    season: str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch unique players with at least one game in the season."""

    def _request() -> pd.DataFrame:
        # LeagueGameLog returns one row per player-game; we de-duplicate to get unique players.
        endpoint = LeagueGameLog(
            season=season,
            # nba_api arg maps to API request param `SeasonTypeAllStar`.
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
        )
        frames = endpoint.get_data_frames()
        if not frames:
            return pd.DataFrame(columns=["player_id", "player_name"])
        frame = frames[0]
        if frame.empty:
            return pd.DataFrame(columns=["player_id", "player_name"])

        # Select only the ID and name columns (other columns are per-game stats we don't need here).
        cols = [c for c in ["PLAYER_ID", "PLAYER_NAME"] if c in frame.columns]
        players = frame[cols].drop_duplicates().rename(
            columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}
        )
        # Coerce player_id to int; rows with non-numeric IDs are dropped.
        players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
        players = players.dropna(subset=["player_id"])
        players["player_id"] = players["player_id"].astype(int)
        players["player_name"] = players["player_name"].fillna("Unknown")
        return players.sort_values("player_name").reset_index(drop=True)

    return _run_with_retry(
        _request,
        op_name=f"LeagueGameLog(season={season}, season_type={season_type})",
    )
