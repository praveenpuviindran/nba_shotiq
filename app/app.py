from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_model_artifact_paths
from src.data.db import initialize_database
from src.data.ingest import ensure_player_shots_cached
from src.modeling.predict import add_p_make, load_metadata, model_artifacts_exist
from src.modeling.train import train_model_for_season
from src.nba.players import get_player_names, resolve_player_id
from src.nba.seasons import get_default_season, get_season_options
from src.nba.shots_api import NbaApiError
from src.viz.heatmaps import (
    plot_frequency_heatmap,
    plot_quality_heatmap,
    plot_smoe_heatmap,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="NBA ShotIQ", layout="wide")
st.title("NBA ShotIQ: Shot Quality vs Shot Making")
st.caption(
    "Explore where players shoot, how difficult those shots are, and whether they outperform expected outcomes."
)
st.info(
    "How to use this page: Pick a player, season, and season type in the sidebar, then click "
    "**Run / Load**. Use the tabs to compare shot volume (Frequency), expected accuracy (Quality), "
    "and over/under-performance (SMOE)."
)

initialize_database()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_player_names() -> list[str]:
    return get_player_names(active_only=False)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def cached_player_shots(
    player_id: int,
    player_name: str,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    return ensure_player_shots_cached(
        player_id=player_id,
        player_name=player_name,
        season=season,
        season_type=season_type,
        force_refresh=False,
    )


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{100 * value:.1f}%"


def _load_model_metadata(season: str, season_type: str) -> dict | None:
    if not model_artifacts_exist(season=season, season_type=season_type):
        return None
    try:
        return load_metadata(season=season, season_type=season_type)
    except Exception:
        return None


def _build_location_table(shots: pd.DataFrame, min_attempts: int = 15) -> pd.DataFrame:
    frame = shots.copy()
    frame["loc_x"] = pd.to_numeric(frame["loc_x"], errors="coerce")
    frame["loc_y"] = pd.to_numeric(frame["loc_y"], errors="coerce")
    frame["shot_distance"] = pd.to_numeric(frame.get("shot_distance"), errors="coerce")
    frame["shot_made_flag"] = pd.to_numeric(frame["shot_made_flag"], errors="coerce")
    frame["p_make"] = pd.to_numeric(frame["p_make"], errors="coerce")
    frame = frame.dropna(subset=["loc_x", "loc_y", "shot_made_flag", "p_make"]).copy()
    if frame.empty:
        return pd.DataFrame()

    fallback_distance = np.sqrt(np.square(frame["loc_x"]) + np.square(frame["loc_y"])) / 10.0
    frame["distance"] = frame["shot_distance"].fillna(fallback_distance)
    frame["angle_deg"] = np.degrees(np.arctan2(frame["loc_y"], frame["loc_x"]))
    frame["made_minus_expected"] = frame["shot_made_flag"] - frame["p_make"]

    distance_bins = [0, 8, 16, 24, 35, 100]
    distance_labels = ["0-8 ft", "8-16 ft", "16-24 ft", "24-35 ft", "35+ ft"]
    angle_bins = [-181, -120, -60, -20, 20, 60, 120, 181]
    angle_labels = ["Far Left", "Left Wing", "Left", "Center", "Right", "Right Wing", "Far Right"]

    frame["distance_bucket"] = pd.cut(
        frame["distance"], bins=distance_bins, labels=distance_labels, right=False, include_lowest=True
    )
    frame["angle_bucket"] = pd.cut(frame["angle_deg"], bins=angle_bins, labels=angle_labels, include_lowest=True)

    grouped = (
        frame.groupby(["distance_bucket", "angle_bucket"], observed=False)
        .agg(
            attempts=("shot_made_flag", "size"),
            fg_pct=("shot_made_flag", "mean"),
            xfg_pct=("p_make", "mean"),
            smoe=("made_minus_expected", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["attempts"] >= min_attempts].copy()
    if grouped.empty:
        return pd.DataFrame()

    grouped["location"] = grouped["distance_bucket"].astype(str) + " | " + grouped["angle_bucket"].astype(str)
    grouped["fg_pct"] = grouped["fg_pct"] * 100
    grouped["xfg_pct"] = grouped["xfg_pct"] * 100
    grouped["smoe"] = grouped["smoe"] * 100

    best = grouped.sort_values("smoe", ascending=False).head(5).copy()
    best["segment"] = "Best Spots"

    worst = grouped.sort_values("smoe", ascending=True).head(5).copy()
    worst["segment"] = "Worst Spots"

    table = pd.concat([best, worst], ignore_index=True)
    table = table[
        ["segment", "location", "attempts", "fg_pct", "xfg_pct", "smoe"]
    ].rename(
        columns={
            "segment": "Group",
            "location": "Location Bucket",
            "attempts": "Attempts",
            "fg_pct": "FG%",
            "xfg_pct": "xFG%",
            "smoe": "SMOE",
        }
    )

    return table.round({"FG%": 1, "xFG%": 1, "SMOE": 1})


with st.sidebar:
    st.header("Controls")
    all_players = cached_player_names()
    default_player = "Stephen Curry" if "Stephen Curry" in all_players else all_players[0]

    player_name = st.selectbox("Player", options=all_players, index=all_players.index(default_player))

    seasons = get_season_options()
    default_season = get_default_season()
    season = st.selectbox("Season", options=seasons, index=seasons.index(default_season))

    season_type = st.selectbox("Season Type", options=["Regular Season", "Playoffs"], index=0)

    run_load = st.button("Run / Load", type="primary")

if run_load:
    st.session_state["selected_player_name"] = player_name
    st.session_state["selected_season"] = season
    st.session_state["selected_season_type"] = season_type

if "selected_player_name" not in st.session_state:
    st.info("Choose options in the sidebar and click **Run / Load**.")
    st.stop()

player_name = st.session_state["selected_player_name"]
season = st.session_state["selected_season"]
season_type = st.session_state["selected_season_type"]

player_id = resolve_player_id(player_name)

model_metadata = _load_model_metadata(season=season, season_type=season_type)
model_ready_for_selected_season = bool(
    model_metadata
    and model_metadata.get("season") == season
    and model_metadata.get("season_type") == season_type
)

if not model_ready_for_selected_season:
    st.warning("No trained model found for this season + season type.")
    st.caption(
        "Train once for this context to unlock expected-make quality and SMOE analytics."
    )
    if st.button("Train model for this season"):
        with st.spinner(
            "Training model. This may take several minutes while fetching and caching season shots..."
        ):
            try:
                train_model_for_season(season=season, season_type=season_type)
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
                st.stop()

        st.cache_data.clear()
        st.success("Model training complete. Click Run / Load to refresh predictions.")
        st.stop()

try:
    with st.spinner("Loading player shots (from cache if available)..."):
        shots = cached_player_shots(
            player_id=player_id,
            player_name=player_name,
            season=season,
            season_type=season_type,
        )
except NbaApiError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to load shot data: {exc}")
    st.stop()

if shots.empty:
    st.warning("No shots found for the selected player/season.")
    st.stop()

shots = shots.copy()
if model_ready_for_selected_season:
    try:
        shots = add_p_make(shots, season=season, season_type=season_type)
    except Exception as exc:
        st.error(f"Unable to score shots with trained model: {exc}")
        st.stop()

shots["shot_made_flag"] = pd.to_numeric(shots["shot_made_flag"], errors="coerce").fillna(0).astype(int)
if "p_make" in shots.columns:
    shots["p_make"] = pd.to_numeric(shots["p_make"], errors="coerce")
else:
    shots["p_make"] = np.nan
shots["made_minus_expected"] = shots["shot_made_flag"] - shots["p_make"]
has_predictions = shots["p_make"].notna().any()

attempts = int(len(shots))
fg_pct = float(shots["shot_made_flag"].mean())

if has_predictions:
    xfg_pct: float | None = float(shots["p_make"].mean())
    smoe_total: float | None = fg_pct - xfg_pct
    shot_diet_difficulty: float | None = 1 - xfg_pct
else:
    xfg_pct = None
    smoe_total = None
    shot_diet_difficulty = None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Attempts", f"{attempts}")
c2.metric("FG%", _format_pct(fg_pct))
c3.metric("xFG%", _format_pct(xfg_pct))
c4.metric("SMOE (FG - xFG)", _format_pct(smoe_total) if smoe_total is not None else "-")
c5.metric(
    "Shot Diet Difficulty",
    _format_pct(shot_diet_difficulty) if shot_diet_difficulty is not None else "-",
    help="Higher means the player took tougher average shots (lower expected make probability).",
)

st.subheader("Chart Guide")
g1, g2, g3 = st.columns(3)
g1.info("**Frequency**: Where the player takes the most shots.")
g2.info("**Quality**: Expected make probability by location (model-based).")
g3.info("**SMOE**: Actual minus expected shooting by location.")

tab_freq, tab_quality, tab_smoe = st.tabs(["Frequency", "Quality", "SMOE"])

with tab_freq:
    fig = plot_frequency_heatmap(shots)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

with tab_quality:
    if not has_predictions:
        st.info("Train/load a model for this season to view quality heatmap.")
    else:
        fig = plot_quality_heatmap(shots)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

with tab_smoe:
    if not has_predictions:
        st.info("Train/load a model for this season to view SMOE heatmap.")
    else:
        fig = plot_smoe_heatmap(shots)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

st.subheader("Best and Worst Spots")
st.caption(
    "Location buckets with at least 15 attempts. SMOE compares actual makes to expected makes."
)
if not has_predictions:
    st.info("Train/load a model for this season to generate best/worst spot tables.")
else:
    spots_table = _build_location_table(shots, min_attempts=15)
    if spots_table.empty:
        st.info("Not enough attempts in any location bucket to summarize (minimum 15 attempts).")
    else:
        st.dataframe(spots_table, use_container_width=True, hide_index=True)

st.subheader("Model Diagnostics")
if model_metadata is None:
    st.write("Model metadata not found.")
else:
    xgb_metrics = model_metadata.get("metrics", {}).get("xgboost", {})
    baseline_metrics = model_metadata.get("metrics", {}).get("baseline_logistic", {})

    md1, md2, md3 = st.columns(3)
    md1.metric("Model Season", model_metadata.get("season", "-"))
    md2.metric("XGB Log Loss", f"{xgb_metrics.get('log_loss', float('nan')):.4f}")
    md3.metric("XGB Brier", f"{xgb_metrics.get('brier_score', float('nan')):.4f}")

    with st.expander("View baseline vs final metrics"):
        st.json({"baseline_logistic": baseline_metrics, "xgboost": xgb_metrics})

    artifact_paths = get_model_artifact_paths(season=season, season_type=season_type)
    calibration_plot_path = artifact_paths["calibration_plot_path"]
    if calibration_plot_path.exists():
        st.image(str(calibration_plot_path), caption="Calibration curve (test split)")
    else:
        st.caption("Calibration plot will appear after training.")
