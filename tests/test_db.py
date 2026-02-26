import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.db import (
    initialize_database,
    read_player_shots,
    season_shots_count,
    upsert_shots,
)


def test_db_schema_and_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "test_nba_shotiq.db"
    initialize_database(db_path)

    rows = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "season_type": "Regular Season",
                "player_id": 201939,
                "player_name": "Stephen Curry",
                "team_id": 1610612744,
                "team_name": "Golden State Warriors",
                "game_id": "0022500001",
                "game_event_id": 10,
                "game_date": "2025-10-21",
                "loc_x": 10.0,
                "loc_y": 40.0,
                "shot_made_flag": 1,
                "shot_distance": 5,
                "shot_type": "2PT Field Goal",
                "action_type": "Driving Layup Shot",
                "shot_zone_basic": "Restricted Area",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "Less Than 8 ft.",
            },
            {
                "season": "2025-26",
                "season_type": "Regular Season",
                "player_id": 201939,
                "player_name": "Stephen Curry",
                "team_id": 1610612744,
                "team_name": "Golden State Warriors",
                "game_id": "0022500001",
                "game_event_id": 11,
                "game_date": "2025-10-21",
                "loc_x": -200.0,
                "loc_y": 200.0,
                "shot_made_flag": 0,
                "shot_distance": 26,
                "shot_type": "3PT Field Goal",
                "action_type": "Jump Shot",
                "shot_zone_basic": "Above the Break 3",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "24+ ft.",
            },
        ]
    )

    inserted = upsert_shots(rows, db_path=db_path)
    assert inserted == 2

    loaded = read_player_shots(
        player_id=201939,
        season="2025-26",
        season_type="Regular Season",
        db_path=db_path,
    )

    assert len(loaded) == 2
    assert season_shots_count("2025-26", "Regular Season", db_path=db_path) == 2

    inserted_again = upsert_shots(rows, db_path=db_path)
    assert inserted_again == 0
