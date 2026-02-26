from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

COURT_X_MIN = -250
COURT_X_MAX = 250
COURT_Y_MIN = -47.5
COURT_Y_MAX = 422.5


# nba_api ShotChartDetail LOC_X/LOC_Y are already in NBA half-court coordinates:
# rim center is near (0, 0), X increases to shooter's right, Y increases toward midcourt.
def transform_shot_coords(loc_x: float, loc_y: float) -> tuple[float, float]:
    return float(loc_x), float(loc_y)


def draw_half_court(ax: plt.Axes, line_color: str = "#3b3b3b", line_width: float = 1.6) -> plt.Axes:
    """Draw an NBA half-court using matplotlib primitives."""
    hoop = Circle((0, 0), radius=7.5, linewidth=line_width, color=line_color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=line_width, color=line_color)

    outer_paint = Rectangle((-80, -47.5), 160, 190, linewidth=line_width, color=line_color, fill=False)
    inner_paint = Rectangle((-60, -47.5), 120, 190, linewidth=line_width, color=line_color, fill=False)

    free_throw_top = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=line_width, color=line_color, fill=False)
    free_throw_bottom = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linestyle="dashed", linewidth=line_width, color=line_color)

    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=line_width, color=line_color)

    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=line_width, color=line_color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=line_width, color=line_color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=line_width, color=line_color)

    center_outer = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=line_width, color=line_color)
    center_inner = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=line_width, color=line_color)

    elements = [
        hoop,
        backboard,
        outer_paint,
        inner_paint,
        free_throw_top,
        free_throw_bottom,
        restricted,
        corner_three_a,
        corner_three_b,
        three_arc,
        center_outer,
        center_inner,
    ]

    for element in elements:
        ax.add_patch(element)

    ax.set_xlim(COURT_X_MIN, COURT_X_MAX)
    ax.set_ylim(COURT_Y_MIN, COURT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax
