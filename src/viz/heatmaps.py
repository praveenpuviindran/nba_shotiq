from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

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


def _partitioned_grid(
    frame: pd.DataFrame,
    value_col: str | None = None,
    x_bins: int = 14,
    y_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(COURT_X_MIN, COURT_X_MAX, x_bins + 1)
    y_edges = np.linspace(COURT_Y_MIN, COURT_Y_MAX, y_bins + 1)

    x = frame["plot_x"].to_numpy()
    y = frame["plot_y"].to_numpy()

    if value_col is None:
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        z = counts.T
        z[z == 0] = np.nan
        return x_edges, y_edges, z

    values = pd.to_numeric(frame[value_col], errors="coerce")
    valid = values.notna().to_numpy()
    if not valid.any():
        z = np.full((y_bins, x_bins), np.nan)
        return x_edges, y_edges, z

    sums, _, _ = np.histogram2d(
        x[valid],
        y[valid],
        bins=[x_edges, y_edges],
        weights=values[valid].to_numpy(),
    )
    counts, _, _ = np.histogram2d(x[valid], y[valid], bins=[x_edges, y_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return x_edges, y_edges, means.T


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
) -> plt.Figure:
    frame = _prepare_plot_frame(shots)
    if frame.empty:
        return _empty_figure(title)
    if value_col is not None and value_col not in frame.columns:
        return _empty_figure(title)

    x_edges, y_edges, z = _partitioned_grid(frame, value_col=value_col)

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)

    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if center_zero else None
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        shading="auto",
        cmap=cmap,
        vmin=None if center_zero else vmin,
        vmax=None if center_zero else vmax,
        norm=norm,
        alpha=0.78,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
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
        title="Shot Frequency Heatmap (Partitioned Zones)",
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
        title="Shot Quality Heatmap (Partitioned Zones)",
        cbar_label="Expected Make Prob",
        value_col="p_make",
        cmap="viridis",
        vmin=0.2,
        vmax=0.8,
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
        title="SMOE Heatmap (Partitioned Zones)",
        cbar_label="Made - Expected",
        value_col="made_minus_expected",
        cmap="RdBu",
        vmin=-0.25,
        vmax=0.25,
        center_zero=True,
    )
