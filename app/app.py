from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CALIBRATION_PLOT_PATH
from src.data.db import initialize_database
from src.data.ingest import ensure_player_shots_cached
from src.modeling.predict import add_p_make, load_metadata, model_artifacts_exist
from src.modeling.train import train_model_for_season
from src.nba.players import get_player_names, resolve_player_id
from src.nba.seasons import get_default_season, get_supported_seasons
from src.nba.shots_api import NbaApiError
from src.viz.heatmaps import (
    plot_frequency_heatmap,
    plot_quality_heatmap,
    plot_smoe_heatmap,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="NBA ShotIQ", layout="wide")
st.title("NBA ShotIQ: Shot Quality vs Shot Making")
st.caption("Single-season MVP using nba_api shot charts, SQLite caching, and an XGBoost xFG model.")

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


def _load_model_metadata() -> dict | None:
    if not model_artifacts_exist():
        return None
    try:
        return load_metadata()
    except Exception:
        return None


with st.sidebar:
    st.header("Controls")
    all_players = cached_player_names()
    default_player = "Stephen Curry" if "Stephen Curry" in all_players else all_players[0]

    player_name = st.selectbox("Player", options=all_players, index=all_players.index(default_player))

    seasons = get_supported_seasons()
    default_season = get_default_season()
    season = st.selectbox("Season", options=seasons, index=seasons.index(default_season))

    season_type = st.selectbox("Season Type", options=["Regular Season"], index=0)

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

model_metadata = _load_model_metadata()
model_ready_for_selected_season = bool(
    model_metadata
    and model_metadata.get("season") == season
    and model_metadata.get("season_type") == season_type
)

if not model_ready_for_selected_season:
    st.warning(
        "No trained model found for this season. Train once to enable Shot Quality and SMOE views."
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
        shots = add_p_make(shots, season=season)
    except Exception as exc:
        st.error(f"Unable to score shots with trained model: {exc}")
        st.stop()

attempts = int(len(shots))
fg_pct = float(pd.to_numeric(shots["shot_made_flag"], errors="coerce").fillna(0).mean())

if "p_make" in shots.columns:
    xfg_pct: float | None = float(pd.to_numeric(shots["p_make"], errors="coerce").mean())
    smoe_total: float | None = fg_pct - xfg_pct
else:
    xfg_pct = None
    smoe_total = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Attempts", f"{attempts}")
c2.metric("FG%", _format_pct(fg_pct))
c3.metric("xFG%", _format_pct(xfg_pct))
c4.metric("SMOE (FG - xFG)", _format_pct(smoe_total) if smoe_total is not None else "-")

tab_freq, tab_quality, tab_smoe = st.tabs(["Frequency", "Quality", "SMOE"])

with tab_freq:
    fig = plot_frequency_heatmap(shots)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

with tab_quality:
    if "p_make" not in shots.columns:
        st.info("Train/load a model for this season to view quality heatmap.")
    else:
        fig = plot_quality_heatmap(shots)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

with tab_smoe:
    if "p_make" not in shots.columns:
        st.info("Train/load a model for this season to view SMOE heatmap.")
    else:
        smoe_df = shots.copy()
        smoe_df["smoe"] = smoe_df["shot_made_flag"] - smoe_df["p_make"]
        fig = plot_smoe_heatmap(smoe_df)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

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

    if CALIBRATION_PLOT_PATH.exists():
        st.image(str(CALIBRATION_PLOT_PATH), caption="Calibration curve (test split)")
    else:
        st.caption("Calibration plot will appear after training.")
