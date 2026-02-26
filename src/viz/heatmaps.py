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
    cbar.set_label("Mean predicted make probability")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_smoe_heatmap(shots: pd.DataFrame) -> plt.Figure:
    title = "Shot Making Over Expected by Location (Mean made - p_make)"
    frame = _prepare_plot_frame(shots)
    if frame.empty or "smoe" not in frame.columns:
        return _empty_figure(title)

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_half_court(ax)

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25)
    hb = ax.hexbin(
        frame["plot_x"],
        frame["plot_y"],
        C=frame["smoe"],
        reduce_C_function=np.mean,
        gridsize=34,
        extent=(COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX),
        mincnt=1,
        cmap="RdBu_r",
        linewidths=0.2,
        norm=norm,
    )
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Mean (made - expected)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
