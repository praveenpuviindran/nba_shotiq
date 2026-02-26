from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle, Wedge

from src.viz.court import (
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    draw_half_court,
    transform_shot_coords,
)


def _prepare_plot_frame(shots: pd.DataFrame) -> pd.DataFrame:
    frame = shots.copy()
    if frame.empty:
        return frame

    frame["loc_x"] = pd.to_numeric(frame["loc_x"], errors="coerce")
    frame["loc_y"] = pd.to_numeric(frame["loc_y"], errors="coerce")
    frame = frame.dropna(subset=["loc_x", "loc_y"]).copy()

    coords = frame.apply(lambda row: transform_shot_coords(row["loc_x"], row["loc_y"]), axis=1)
    frame["plot_x"] = [c[0] for c in coords]
    frame["plot_y"] = [c[1] for c in coords]
    return frame


def _empty_figure(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)
    ax.text(0, 180, "No shot data available", ha="center", va="center")
    ax.set_title(title)
    return fig


def _zone_id(x: float, y: float) -> str | None:
    if x < COURT_X_MIN or x > COURT_X_MAX or y < COURT_Y_MIN or y > COURT_Y_MAX:
        return None

    if y <= 140 and x <= -220:
        return "corner3_left"
    if y <= 140 and x >= 220:
        return "corner3_right"

    distance = float(np.hypot(x, y))
    theta = float(np.degrees(np.arctan2(abs(y), x)))

    if distance <= 80:
        return "restricted"

    if y <= 190 and abs(x) <= 80:
        if x < -26.67:
            return "paint_left"
        if x > 26.67:
            return "paint_right"
        return "paint_center"

    if distance <= 237.5:
        if theta >= 140:
            return "mid_left"
        if theta >= 110:
            return "mid_left_center"
        if theta >= 70:
            return "mid_center"
        if theta >= 40:
            return "mid_right_center"
        return "mid_right"

    if theta >= 120:
        return "three_left"
    if theta >= 60:
        return "three_center"
    return "three_right"


def _zone_patches() -> dict[str, object]:
    paint_width = 160.0 / 3.0

    return {
        "corner3_left": Rectangle((-250, -47.5), 30, 187.5),
        "corner3_right": Rectangle((220, -47.5), 30, 187.5),
        "restricted": Circle((0, 0), 80),
        "paint_left": Rectangle((-80, -47.5), paint_width, 237.5),
        "paint_center": Rectangle((-80 + paint_width, -47.5), paint_width, 237.5),
        "paint_right": Rectangle((-80 + 2 * paint_width, -47.5), paint_width, 237.5),
        "mid_right": Wedge((0, 0), 237.5, 0, 40, width=157.5),
        "mid_right_center": Wedge((0, 0), 237.5, 40, 70, width=157.5),
        "mid_center": Wedge((0, 0), 237.5, 70, 110, width=157.5),
        "mid_left_center": Wedge((0, 0), 237.5, 110, 140, width=157.5),
        "mid_left": Wedge((0, 0), 237.5, 140, 180, width=157.5),
        "three_right": Wedge((0, 0), 320, 0, 60, width=82.5),
        "three_center": Wedge((0, 0), 320, 60, 120, width=82.5),
        "three_left": Wedge((0, 0), 320, 120, 180, width=82.5),
    }


def _zone_values(
    frame: pd.DataFrame,
    value_col: str | None,
    min_attempts: int,
) -> dict[str, float]:
    working = frame.copy()
    working["zone_id"] = working.apply(
        lambda row: _zone_id(float(row["plot_x"]), float(row["plot_y"])),
        axis=1,
    )
    working = working.dropna(subset=["zone_id"]).copy()
    if working.empty:
        return {}

    counts = working.groupby("zone_id").size().to_dict()

    if value_col is None:
        return {zone_id: float(count) for zone_id, count in counts.items()}

    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[value_col]).copy()
    if working.empty:
        return {}

    means = working.groupby("zone_id")[value_col].mean().to_dict()
    return {
        zone_id: float(value)
        for zone_id, value in means.items()
        if int(counts.get(zone_id, 0)) >= min_attempts
    }


def _plot_partitioned_metric(
    shots: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    value_col: str | None = None,
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
    min_attempts: int = 1,
) -> plt.Figure:
    frame = _prepare_plot_frame(shots)
    if frame.empty:
        return _empty_figure(title)
    if value_col is not None and value_col not in frame.columns:
        return _empty_figure(title)

    values = _zone_values(frame, value_col=value_col, min_attempts=min_attempts)
    if not values:
        return _empty_figure(title)

    raw_vals = np.array(list(values.values()), dtype=float)

    cmap_obj = plt.get_cmap(cmap)
    if center_zero:
        max_abs = float(max(abs(np.nanmin(raw_vals)), abs(np.nanmax(raw_vals)), 1e-6))
        norm = TwoSlopeNorm(
            vcenter=0.0,
            vmin=(-max_abs if vmin is None else vmin),
            vmax=(max_abs if vmax is None else vmax),
        )
    else:
        local_vmin = float(np.nanmin(raw_vals)) if vmin is None else vmin
        local_vmax = float(np.nanmax(raw_vals)) if vmax is None else vmax
        if np.isclose(local_vmin, local_vmax):
            local_vmax = local_vmin + 1e-6
        norm = Normalize(vmin=local_vmin, vmax=local_vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#efefef")

    for zone_name, patch in _zone_patches().items():
        zone_value = values.get(zone_name)
        facecolor = "#9f9f9f" if zone_value is None else cmap_obj(norm(zone_value))
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("#ffffff")
        patch.set_linewidth(2.0)
        patch.set_alpha(0.92)
        ax.add_patch(patch)

    draw_half_court(ax, line_color="#1f1f1f", line_width=1.7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_frequency_heatmap(shots: pd.DataFrame) -> plt.Figure:
    title = "Shot Frequency Heatmap (Hexbin Density)"
    frame = _prepare_plot_frame(shots)
    if frame.empty:
        return _empty_figure(title)

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)

    hb = ax.hexbin(
        frame["plot_x"],
        frame["plot_y"],
        gridsize=34,
        extent=(COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX),
        mincnt=1,
        cmap="YlOrRd",
        linewidths=0.2,
    )
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Attempts")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_frequency_partitioned_heatmap(shots: pd.DataFrame) -> plt.Figure:
    return _plot_partitioned_metric(
        shots,
        title="Shot Frequency Heatmap (Court Zones)",
        cbar_label="Attempts",
        value_col=None,
        cmap="YlOrRd",
    )


def plot_quality_heatmap(shots: pd.DataFrame) -> plt.Figure:
    title = "Shot Quality Heatmap (Mean Predicted P(make))"
    frame = _prepare_plot_frame(shots)
    if frame.empty or "p_make" not in frame.columns:
        return _empty_figure(title)

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)

    hb = ax.hexbin(
        frame["plot_x"],
        frame["plot_y"],
        C=frame["p_make"],
        reduce_C_function=np.mean,
        gridsize=34,
        extent=(COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX),
        mincnt=1,
        cmap="viridis",
        linewidths=0.2,
        vmin=0.2,
        vmax=0.8,
    )
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Expected Make Prob")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_quality_partitioned_heatmap(shots: pd.DataFrame) -> plt.Figure:
    return _plot_partitioned_metric(
        shots,
        title="Shot Quality Heatmap (Court Zones)",
        cbar_label="Expected Make Prob",
        value_col="p_make",
        cmap="viridis",
        vmin=0.2,
        vmax=0.8,
        min_attempts=6,
    )


def plot_smoe_heatmap(shots: pd.DataFrame) -> plt.Figure:
    title = "Shot Making Over Expected by Location (Mean made - p_make)"
    frame = _prepare_plot_frame(shots)
    if frame.empty or "made_minus_expected" not in frame.columns:
        return _empty_figure(title)

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25)
    hb = ax.hexbin(
        frame["plot_x"],
        frame["plot_y"],
        C=frame["made_minus_expected"],
        reduce_C_function=np.mean,
        gridsize=34,
        extent=(COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX),
        mincnt=1,
        cmap="RdBu",
        linewidths=0.2,
        norm=norm,
    )
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Made - Expected")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_smoe_partitioned_heatmap(shots: pd.DataFrame) -> plt.Figure:
    return _plot_partitioned_metric(
        shots,
        title="SMOE Heatmap (Court Zones)",
        cbar_label="Made - Expected",
        value_col="made_minus_expected",
        cmap="RdBu",
        center_zero=True,
        min_attempts=6,
    )
