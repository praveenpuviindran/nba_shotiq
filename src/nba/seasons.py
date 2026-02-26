from __future__ import annotations

from src.config import DEFAULT_SEASON

SUPPORTED_SEASONS = [
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
]


def get_supported_seasons() -> list[str]:
    """Return season options for the Streamlit sidebar."""
    return SUPPORTED_SEASONS


def get_default_season() -> str:
    """Return the MVP default season."""
    if DEFAULT_SEASON in SUPPORTED_SEASONS:
        return DEFAULT_SEASON
    return SUPPORTED_SEASONS[0]
