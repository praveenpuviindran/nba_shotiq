from __future__ import annotations

from datetime import date

from src.config import DEFAULT_SEASON

EARLIEST_SEASON_START_YEAR = 1996


def _current_season_start_year(today: date | None = None) -> int:
    current = today or date.today()
    return current.year if current.month >= 9 else current.year - 1


def _format_season(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def get_season_options(today: date | None = None) -> list[str]:
    """
    Generate supported seasons dynamically up to the current NBA season.

    Example output:
    ["1996-97", ..., "2025-26"]
    """
    current_start_year = _current_season_start_year(today=today)
    return [
        _format_season(start_year)
        for start_year in range(EARLIEST_SEASON_START_YEAR, current_start_year + 1)
    ]


def get_default_season() -> str:
    """Return the MVP default season."""
    options = get_season_options()
    if DEFAULT_SEASON in options:
        return DEFAULT_SEASON
    return options[-1]
