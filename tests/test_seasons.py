import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.nba.seasons import get_season_options


def test_season_options_include_2025_26() -> None:
    options = get_season_options(today=date(2026, 2, 26))
    assert options[0] == "1996-97"
    assert options[-1] == "2025-26"
    assert "2025-26" in options


def test_season_rollover_in_september() -> None:
    options = get_season_options(today=date(2026, 9, 2))
    assert options[-1] == "2026-27"
