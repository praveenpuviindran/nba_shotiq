from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.data.db import (
    delete_player_shots,
    initialize_database,
    player_shots_exist,
    read_player_shots,
    read_season_shots,
    season_shots_count,
    upsert_shots,
)
from src.nba.shots_api import NbaApiError, fetch_player_shots, fetch_players_with_games

logger = logging.getLogger(__name__)


def ensure_player_shots_cached(
    player_id: int,
    player_name: str,
    season: str,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load player shots from cache, fetching from API only when missing."""
    initialize_database()

    if force_refresh:
        deleted = delete_player_shots(player_id, season, season_type)
        logger.info("Deleted %s rows for refresh of player_id=%s", deleted, player_id)

    cached = read_player_shots(player_id, season, season_type)
    if not cached.empty:
        return cached

    fetched = fetch_player_shots(
        player_id=player_id,
        player_name=player_name,
        season=season,
        season_type=season_type,
    )
    if fetched.empty:
        logger.warning("No shots returned for %s in season %s", player_name, season)
        return fetched

    upsert_shots(fetched)
    return read_player_shots(player_id, season, season_type)


def ensure_season_shots_cached(
    season: str,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
    max_players: Optional[int] = None,
) -> pd.DataFrame:
    """Ensure the cache contains shot data for all players with games in a season."""
    initialize_database()

    if not force_refresh and season_shots_count(season, season_type) > 0:
        logger.info("Season cache already populated for %s (%s)", season, season_type)
        return read_season_shots(season, season_type)

    players = fetch_players_with_games(season=season, season_type=season_type)
    if players.empty:
        raise NbaApiError(
            f"No players were returned for season={season}, season_type={season_type}."
        )

    if max_players is not None:
        players = players.head(max_players)

    total = len(players)
    logger.info("Caching shot charts for %s players (%s, %s)", total, season, season_type)

    for idx, row in players.iterrows():
        player_id = int(row["player_id"])
        player_name = str(row["player_name"])

        if not force_refresh and player_shots_exist(player_id, season, season_type):
            continue

        try:
            ensure_player_shots_cached(
                player_id=player_id,
                player_name=player_name,
                season=season,
                season_type=season_type,
                force_refresh=force_refresh,
            )
        except NbaApiError as exc:
            logger.warning(
                "Skipping player_id=%s (%s) due to API error: %s",
                player_id,
                player_name,
                exc,
            )

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            logger.info("Processed %s/%s players", idx + 1, total)

    return read_season_shots(season, season_type)
