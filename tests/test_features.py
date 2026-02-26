import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.features import add_engineered_features, infer_is_three_point


def test_distance_and_angle_computation() -> None:
    df = pd.DataFrame(
        {
            "loc_x": [3.0, -3.0],
            "loc_y": [4.0, 4.0],
            "shot_distance": [5.0, 24.0],
            "shot_type": ["2PT Field Goal", "3PT Field Goal"],
        }
    )

    out = add_engineered_features(df)

    assert np.isclose(out.loc[0, "distance"], 5.0)
    assert np.isclose(out.loc[0, "angle"], np.arctan2(4.0, 3.0))
    assert np.isclose(out.loc[1, "angle"], np.arctan2(4.0, -3.0))


def test_is_three_point_inference() -> None:
    df = pd.DataFrame(
        {
            "loc_x": [0.0, 230.0, 0.0],
            "loc_y": [0.0, 60.0, 0.0],
            "shot_distance": [10.0, 22.0, 25.0],
            "shot_type": ["2PT Field Goal", "2PT Field Goal", "2PT Field Goal"],
            "shot_zone_basic": ["Mid-Range", "Right Corner 3", "Above the Break 3"],
            "shot_zone_range": ["8-16 ft.", "24+ ft.", "24+ ft."],
        }
    )

    is_three = infer_is_three_point(df)
    assert list(is_three) == [0, 1, 1]
