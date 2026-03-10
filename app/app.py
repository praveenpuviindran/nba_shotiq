from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    GITHUB_REPO_URL,
    TABLEAU_EMBED_URL,
    TABLEAU_PUBLIC_VIEW_URL,
    get_model_artifact_paths,
)
from src.data.db import initialize_database, read_season_shots
from src.data.ingest import ensure_player_shots_cached, ensure_season_shots_cached
from src.modeling.predict import add_p_make, load_metadata, model_artifacts_exist
from src.modeling.train import train_model_for_season
from src.nba.players import get_player_names, resolve_player_id
from src.nba.seasons import get_default_season, get_season_options
from src.nba.shots_api import NbaApiError
from src.viz.heatmaps import (
    plot_frequency_heatmap,
    plot_frequency_partitioned_heatmap,
    plot_quality_heatmap,
    plot_quality_partitioned_heatmap,
    plot_smoe_heatmap,
    plot_smoe_partitioned_heatmap,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="NBA ShotIQ", layout="wide")
st.title("NBA ShotIQ")
st.caption("A simple way to understand where NBA players shoot, how hard those shots are, and how they perform.")
st.info(
    "Use the left sidebar to switch between the player dashboard, Tableau public dashboard, "
    "and Tableau export workflow."
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


@st.cache_data(ttl=30 * 60, show_spinner=False)
def cached_season_player_metrics(
    season: str,
    season_type: str,
    use_full_cache: bool,
) -> pd.DataFrame:
    if use_full_cache:
        season_shots = ensure_season_shots_cached(
            season=season,
            season_type=season_type,
            force_refresh=False,
        )
    else:
        season_shots = read_season_shots(season=season, season_type=season_type)

    if season_shots.empty:
        return pd.DataFrame()

    scored = add_p_make(season_shots, season=season, season_type=season_type)
    scored["shot_made_flag"] = pd.to_numeric(scored["shot_made_flag"], errors="coerce").fillna(0).astype(int)
    scored["p_make"] = pd.to_numeric(scored["p_make"], errors="coerce")
    scored["made_minus_expected"] = scored["shot_made_flag"] - scored["p_make"]

    metrics = (
        scored.groupby(["player_id", "player_name"], as_index=False)
        .agg(
            Attempts=("shot_made_flag", "size"),
            FG_pct=("shot_made_flag", "mean"),
            xFG_pct=("p_make", "mean"),
            SMOE_raw=("made_minus_expected", "mean"),
        )
        .sort_values("Attempts", ascending=False)
    )

    metrics["SMOE_raw"] = metrics["FG_pct"] - metrics["xFG_pct"]
    metrics["Shot_Diet_Difficulty"] = 1 - metrics["xFG_pct"]

    metrics["FG%"] = metrics["FG_pct"] * 100
    metrics["xFG%"] = metrics["xFG_pct"] * 100
    metrics["SMOE"] = metrics["SMOE_raw"] * 100
    metrics["Shot Diet Difficulty"] = metrics["Shot_Diet_Difficulty"] * 100

    return metrics[
        [
            "player_id",
            "player_name",
            "Attempts",
            "FG%",
            "xFG%",
            "SMOE",
            "Shot Diet Difficulty",
        ]
    ]


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


def _render_shot_map_views(
    shots: pd.DataFrame,
    view_style: str,
    hexbin_fn,
    partitioned_fn,
) -> None:
    if view_style == "Hexbin":
        st.pyplot(hexbin_fn(shots), clear_figure=True, use_container_width=True)
        return

    if view_style == "Court Zones":
        st.pyplot(partitioned_fn(shots), clear_figure=True, use_container_width=True)
        return

    left, right = st.columns(2)
    with left:
        st.caption("Hexbin")
        st.pyplot(hexbin_fn(shots), clear_figure=True, use_container_width=True)
    with right:
        st.caption("Court Zones")
        st.pyplot(partitioned_fn(shots), clear_figure=True, use_container_width=True)


def _render_tableau_page() -> None:
    st.subheader("Interactive Dashboard")
    st.caption(
        "This executive view summarizes shot quality versus shot results with KPI cards, "
        "a shot chart, a shot-quality-vs-efficiency scatter, and a player leaderboard."
    )
    b1, b2 = st.columns(2)
    b1.link_button("Open on Tableau Public", TABLEAU_PUBLIC_VIEW_URL, use_container_width=True)
    b2.link_button("Open GitHub Repo", GITHUB_REPO_URL, use_container_width=True)

    try:
        components.iframe(TABLEAU_EMBED_URL, height=960, scrolling=True)
    except Exception as exc:
        st.warning(f"Could not embed Tableau in-app: {exc}")
        st.markdown(f"[Open Tableau Public dashboard directly]({TABLEAU_PUBLIC_VIEW_URL})")


def _run_tableau_export(
    season: str,
    season_type: str,
    out_dir: Path,
    fetch_if_missing: bool,
    force_refresh: bool,
) -> tuple[bool, str, list[str]]:
    script_path = PROJECT_ROOT / "scripts" / "export_tableau.py"
    if not script_path.exists():
        return False, f"Export script not found: {script_path}", []

    cmd = [
        sys.executable,
        str(script_path),
        "--season",
        season,
        "--season-type",
        season_type,
        "--out",
        str(out_dir),
    ]
    if not fetch_if_missing:
        cmd.append("--no-fetch-if-missing")
    if force_refresh:
        cmd.append("--force-refresh")

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logs = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return completed.returncode == 0, logs, cmd


def _render_export_files_table(out_dir: Path) -> None:
    files = sorted(out_dir.glob("*.csv"))
    if not files:
        st.info(f"No CSV exports found yet in `{out_dir}`.")
        return

    frame = pd.DataFrame(
        [
            {
                "file": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
            for path in files
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_exports_page() -> None:
    st.subheader("Data Export for Tableau")
    st.caption(
        "Generate Tableau-ready CSVs from local SQLite cache. "
        "Use API fetch only when your selected season is not cached yet."
    )
    seasons = get_season_options()
    season_start_years = sorted({int(season[:4]) for season in seasons}, reverse=True)
    default_start_year = int(get_default_season()[:4])
    selected_start_year = st.selectbox(
        "Season (start year)",
        options=season_start_years,
        index=season_start_years.index(default_start_year) if default_start_year in season_start_years else 0,
    )
    season_type = st.selectbox("Season type", options=["Regular Season", "Playoffs"], index=0)
    fetch_if_missing = st.checkbox(
        "Fetch from nba_api when season cache is missing (slower)",
        value=False,
    )
    force_refresh = st.checkbox("Force refresh season from nba_api (slowest)", value=False)
    if force_refresh:
        st.warning("Force refresh bypasses cache and can take a long time for full-season pulls.")

    out_dir = PROJECT_ROOT / "data" / "tableau_exports"
    if st.button("Generate Tableau CSV Exports", type="primary"):
        with st.spinner("Running Tableau export..."):
            success, logs, cmd = _run_tableau_export(
                season=str(selected_start_year),
                season_type=season_type,
                out_dir=out_dir,
                fetch_if_missing=fetch_if_missing,
                force_refresh=force_refresh,
            )
        st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")
        if logs:
            st.code(logs, language="text")
        if success:
            st.success(f"Exports written to `{out_dir}`")
        else:
            st.error("Export failed. Review logs above.")

    st.markdown(f"Output directory: `{out_dir}`")
    _render_export_files_table(out_dir)


def _render_model_health_page() -> None:
    import json
    import plotly.express as px
    from monitoring.monitor import ModelMonitor, initialize_monitoring_tables
    from src.config import DB_PATH

    st.subheader("Model Health")
    st.caption(
        "Tracks calibration quality (Brier score, ECE, log loss) and feature "
        "distribution drift (PSI) across a rolling 30-day window."
    )

    initialize_monitoring_tables(DB_PATH)

    history_path = PROJECT_ROOT / "monitoring" / "monitoring_history.json"
    history: list[dict] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []

    latest = history[-1] if history else None

    if latest and latest.get("alert_triggered"):
        st.error(f"Alert: {latest.get('alert_message', 'Check monitoring report.')}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brier Score", f"{latest['brier_score']:.4f}" if latest and latest.get("brier_score") else "—")
    m2.metric("ECE", f"{latest['ece']:.4f}" if latest and latest.get("ece") else "—")
    m3.metric("Log Loss", f"{latest['log_loss']:.4f}" if latest and latest.get("log_loss") else "—")
    m4.metric("Last Run", latest["timestamp"][:10] if latest else "Never")

    if len(history) >= 2:
        hist_df = pd.DataFrame(history)
        hist_df["run_date"] = pd.to_datetime(hist_df["timestamp"]).dt.date
        hist_df = hist_df.dropna(subset=["brier_score"])
        if not hist_df.empty:
            fig = px.line(
                hist_df,
                x="run_date",
                y="brier_score",
                title="Brier Score Over Time",
                markers=True,
                labels={"run_date": "Date", "brier_score": "Brier Score"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if latest and latest.get("drift_report"):
        st.subheader("Feature Drift (PSI)")
        drift_rows = [
            {"Feature": feat, "PSI": info["psi"], "Status": info["status"]}
            for feat, info in latest["drift_report"].items()
        ]
        drift_df = pd.DataFrame(drift_rows)
        color_map = {"ok": "#22c55e", "warn": "#f59e0b", "alert": "#ef4444"}
        drift_df["Color"] = drift_df["Status"].map(color_map)
        fig_psi = px.bar(
            drift_df,
            x="Feature",
            y="PSI",
            color="Status",
            color_discrete_map={"ok": "#22c55e", "warn": "#f59e0b", "alert": "#ef4444"},
            title="Population Stability Index per Feature",
        )
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Warn threshold")
        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Alert threshold")
        st.plotly_chart(fig_psi, use_container_width=True)
    else:
        st.info("No drift data yet. Run a monitoring check to populate the report.")

    st.divider()
    if st.button("Run Monitoring Check", type="primary"):
        with st.spinner("Computing metrics…"):
            monitor = ModelMonitor(model=None, db_path=DB_PATH, window_days=30)
            report = monitor.run_monitoring_report()
        if report.alert_triggered:
            st.error(f"Alert triggered: {report.alert_message}")
        else:
            st.success("All checks passed.")
        st.json({
            "brier_score": report.brier_score,
            "ece": report.ece,
            "log_loss": report.log_loss,
            "n_predictions": report.n_predictions,
            "alert": report.alert_triggered,
        })
        st.rerun()

    st.caption(
        "To establish the baseline (run once): `python monitoring/establish_baseline.py`"
    )


_SQL_INTERPRETATIONS = {
    "01_player_shot_quality_trends": (
        "This query tracks each player's shot selection quality month-by-month within a season. "
        "The average xFG% column shows the expected difficulty of a player's shot diet — "
        "a rising trend means the player is taking progressively easier shots. "
        "The SMOE column (actual − expected) indicates whether the player is outperforming "
        "or underperforming the model's expectations in each month."
    ),
    "02_shot_context_cohort_analysis": (
        "This query groups shots into distance and angle cohorts and compares actual FG% "
        "against model-expected FG% within each court zone. "
        "The SMOE (shot making over expected) column reveals which zone-and-angle combinations "
        "produce the most consistent overperformance or underperformance across all players. "
        "Cohorts ranked highest in SMOE within their zone are where players collectively beat the model."
    ),
    "03_player_overperformance_leaderboard": (
        "This leaderboard ranks players by their SMOE — how much they outperform the model's "
        "expected make rate. Critically, only players whose lower 95% confidence interval bound "
        "is above zero are included, filtering out overperformers whose advantage may be "
        "statistical noise. Players appearing here are consistently beating expectations "
        "across a statistically meaningful sample of shots."
    ),
    "04_game_flow_shot_selection": (
        "This query analyzes how shot selection patterns have evolved across seasons and court zones. "
        "The rolling 2-season average xFG helps smooth out single-season noise. "
        "Zones with rising three_pt_rate alongside stable SMOE suggest a sustainable shift "
        "toward higher-value shot creation, while falling xFG with negative SMOE flags "
        "zones where shot quality has deteriorated."
    ),
}


def _render_sql_analytics_page() -> None:
    import plotly.express as px
    from sql.runner import list_queries, run_query
    from src.config import DB_PATH

    st.subheader("SQL Analytics")
    st.caption(
        "Advanced analytical queries using window functions, CTEs, and cohort analysis. "
        "All queries run directly against the local SQLite database."
    )

    queries = list_queries(PROJECT_ROOT / "sql")
    if not queries:
        st.error("No .sql files found in sql/ directory.")
        return

    query_labels = [q["label"] for q in queries]
    selected_idx = st.selectbox(
        "Select a query:",
        options=range(len(queries)),
        format_func=lambda i: f"{i + 1}. {query_labels[i]}",
    )
    selected_query = queries[selected_idx]

    st.caption(f"File: `{selected_query['file']}`")

    with st.expander("View SQL"):
        sql_text = Path(selected_query["file"]).read_text()
        st.code(sql_text, language="sql")

    if st.button("Run Query", type="primary"):
        with st.spinner("Executing query…"):
            try:
                df = run_query(selected_query["file"], DB_PATH)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                return

        if df.empty:
            st.warning(
                "Query returned no rows. This usually means the predictions table has not been "
                "populated yet. Run `python monitoring/establish_baseline.py` first."
            )
            return

        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode()
        st.download_button(
            "Export CSV",
            data=csv_data,
            file_name=f"{Path(selected_query['file']).stem}.csv",
            mime="text/csv",
        )

        # Plain-English interpretation
        stem = Path(selected_query["file"]).stem
        interpretation = _SQL_INTERPRETATIONS.get(stem)
        if interpretation:
            st.info(interpretation)


def _render_causal_analysis_page() -> None:
    import plotly.graph_objects as go
    import plotly.express as px
    from causal.shot_selection_analysis import (
        load_player_season_data,
        compute_propensity_scores,
        estimate_ipw_effect,
        difference_in_differences,
        sensitivity_analysis,
    )
    from src.config import DB_PATH

    st.subheader("Causal Analysis: Does Shot Selection Cause Efficiency?")
    st.info(
        "This analysis uses causal inference methods, not correlation. "
        "The estimates attempt to control for pre-existing player skill."
    )

    st.markdown("""
### Research Question
**Does improving shot selection quality (taking shots the model rates as higher-probability makes)
cause a player to score more points per shot — beyond what their pre-existing skill would predict?**

This matters for coaching: if the relationship is causal, enforcing shot discipline should produce
measurable scoring gains.  If purely correlational, better shooters simply attract better looks.
""")

    with st.spinner("Loading player-season data…"):
        panel = load_player_season_data(DB_PATH)

    if panel.empty:
        st.warning(
            "No predictions data found. Run `python monitoring/establish_baseline.py` "
            "to populate the predictions table, then return to this page."
        )
        return

    # ── IPW ──────────────────────────────────────────────────────────────── #
    st.divider()
    st.markdown("### Part 1 — Inverse Probability Weighting (IPW)")
    st.caption(
        "Treatment: top-tercile shot selection quality (avg xFG% ≥ 66th percentile). "
        "Outcome: actual points per shot. Confounders: prior FG%, prior xFG%, volume, "
        "3PT rate, shot distance."
    )

    with st.spinner("Estimating propensity scores…"):
        df_ps = compute_propensity_scores(panel)
        ipw = estimate_ipw_effect(df_ps)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATE (pts/shot)", f"{ipw['ate']:+.4f}")
    c2.metric("95% CI", f"[{ipw['ci_lower']:.4f}, {ipw['ci_upper']:.4f}]")
    c3.metric("p-value", f"{ipw['p_value']:.3f}")
    c4.metric("Treated / Control", f"{ipw['n_treated']} / {ipw['n_control']}")

    # Forest plot
    fig_forest = go.Figure()
    fig_forest.add_trace(go.Scatter(
        x=[ipw["ate"]],
        y=["IPW ATE"],
        error_x=dict(
            type="data",
            symmetric=False,
            plus=[ipw["ci_upper"] - ipw["ate"]],
            minus=[ipw["ate"] - ipw["ci_lower"]],
        ),
        mode="markers",
        marker=dict(size=14, color="#3b82f6"),
        name="ATE estimate",
    ))
    fig_forest.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_forest.update_layout(
        title="Average Treatment Effect — Forest Plot",
        xaxis_title="Effect on Points Per Shot",
        yaxis_title="",
        height=250,
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # ── DiD ──────────────────────────────────────────────────────────────── #
    st.divider()
    st.markdown("### Part 2 — Difference-in-Differences (DiD)")
    st.caption(
        "Compares players who improved shot selection (treatment) vs. stable players (control) "
        "across consecutive seasons.  Estimates: δ = (treated_post − treated_pre) − (control_post − control_pre)."
    )

    with st.spinner("Running DiD…"):
        did = difference_in_differences(panel)

    d1, d2, d3 = st.columns(3)
    d1.metric("DiD Estimate (pts/shot)", f"{did['did_estimate']:+.4f}" if did.get("did_estimate") else "—")
    d2.metric("95% CI", f"[{did['ci_lower']:.4f}, {did['ci_upper']:.4f}]" if did.get("ci_lower") else "—")
    d3.metric("p-value", f"{did['p_value']:.3f}" if did.get("p_value") else "—")

    pt = did.get("parallel_trends_data", {})
    if pt and len(pt.get("group", [])) == 2:
        fig_did = go.Figure()
        for i, grp in enumerate(pt["group"]):
            fig_did.add_trace(go.Scatter(
                x=["Pre (season T)", "Post (season T+1)"],
                y=[pt["pre_outcome"][i], pt["post_outcome"][i]],
                mode="lines+markers",
                name=grp,
                marker=dict(size=10),
            ))
        fig_did.update_layout(
            title="DiD — Treated vs Control Group Trends",
            yaxis_title="Points Per Shot",
            xaxis_title="Period",
            height=350,
        )
        st.plotly_chart(fig_did, use_container_width=True)

    st.caption("Parallel trends assumption: both groups should follow similar pre-period trajectories.")

    # ── Rosenbaum ────────────────────────────────────────────────────────── #
    st.divider()
    st.markdown("### Part 3 — Rosenbaum Sensitivity Analysis")
    st.caption(
        "Tests how large an unmeasured confounder would need to be (Γ = odds ratio of "
        "treatment assignment) to explain away the observed effect."
    )

    with st.spinner("Running sensitivity analysis…"):
        sens = sensitivity_analysis(df_ps)

    st.markdown(f"**{sens['interpretation']}**")

    if sens["gamma_table"]:
        sens_df = pd.DataFrame(sens["gamma_table"])
        fig_sens = px.line(
            sens_df,
            x="gamma",
            y="p_upper",
            title="Rosenbaum Bounds — p-value upper bound by Γ",
            markers=True,
            labels={"gamma": "Γ (confounder strength)", "p_upper": "p-value upper bound"},
        )
        fig_sens.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="α = 0.05")
        st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown(
        "Full methodology and limitations: [causal/CAUSAL_FINDINGS.md]"
        f"({GITHUB_REPO_URL}/blob/main/causal/CAUSAL_FINDINGS.md)"
    )


with st.sidebar:
    app_section = st.radio(
        "Navigation",
        options=[
            "Player Dashboard",
            "Tableau Dashboard",
            "Data Exports",
            "Model Health",
            "SQL Analytics",
            "Causal Analysis",
        ],
        index=0,
    )

if app_section == "Tableau Dashboard":
    _render_tableau_page()
    st.stop()

if app_section == "Data Exports":
    _render_exports_page()
    st.stop()

if app_section == "Model Health":
    _render_model_health_page()
    st.stop()

if app_section == "SQL Analytics":
    _render_sql_analytics_page()
    st.stop()

if app_section == "Causal Analysis":
    _render_causal_analysis_page()
    st.stop()

with st.expander("New to basketball stats? Quick glossary"):
    st.markdown(
        """
        - **FG% (Field Goal %)**: Share of shots that went in.
        - **xFG% (Expected FG%)**: Model estimate of how often those shots should go in.
        - **SMOE (Shot Making Over Expected)**: `FG% - xFG%`.
        - **Shot Diet Difficulty**: `1 - xFG%`. Higher means tougher average shots.
        """
    )

with st.sidebar:
    st.header("NBA ShotIQ")
    st.caption("Choose your inputs")
    all_players = cached_player_names()
    default_player = "Stephen Curry" if "Stephen Curry" in all_players else all_players[0]

    player_name = st.selectbox("Player", options=all_players, index=all_players.index(default_player))

    seasons = get_season_options()
    default_season = get_default_season()
    season = st.selectbox("Season", options=seasons, index=seasons.index(default_season))

    season_type = st.selectbox("Game Type", options=["Regular Season", "Playoffs"], index=0)

    run_load = st.button("Load Dashboard", type="primary")

if run_load:
    st.session_state["selected_player_name"] = player_name
    st.session_state["selected_season"] = season
    st.session_state["selected_season_type"] = season_type

if "selected_player_name" not in st.session_state:
    st.info("Choose options in the sidebar and click **Load Dashboard**.")
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
        st.success("Model training complete. Click Load Dashboard to refresh predictions.")
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
c1.metric("Shot Attempts", f"{attempts}", help="Total shots in the selected season and game type.")
c2.metric("Field Goal % (FG%)", _format_pct(fg_pct), help="How often shots were made.")
c3.metric(
    "Expected FG% (xFG%)",
    _format_pct(xfg_pct),
    help="Model estimate of make rate based on shot location and shot profile.",
)
c4.metric(
    "Over/Under Expected (SMOE)",
    _format_pct(smoe_total) if smoe_total is not None else "-",
    help="FG% minus xFG%. Positive means better than expected; negative means below expected.",
)
c5.metric(
    "Shot Diet Difficulty",
    _format_pct(shot_diet_difficulty) if shot_diet_difficulty is not None else "-",
    help="Higher means the player took tougher average shots (lower expected make probability).",
)

st.subheader("Season Leaderboard")
st.caption(
    "Compare players for this season and game type. Sort by any key metric and apply a minimum-attempts filter."
)

lb1, lb2, lb3, lb4 = st.columns(4)
sort_by = lb1.selectbox(
    "Sort by",
    options=["Attempts", "FG%", "xFG%", "SMOE", "Shot Diet Difficulty"],
    index=0,
)
sort_order = lb2.selectbox("Order", options=["Descending", "Ascending"], index=0)
min_attempts = lb3.number_input("Min attempts", min_value=1, max_value=500, value=100, step=5)
top_n = lb4.number_input("Rows", min_value=5, max_value=200, value=25, step=5)

use_full_cache = st.checkbox(
    "Include all players (downloads missing players the first time, which can take longer)",
    value=False,
)

if not model_ready_for_selected_season:
    st.info("Train/load a model for this season + season type to unlock full leaderboard metrics.")
else:
    try:
        with st.spinner("Computing season leaderboard..."):
            leaderboard = cached_season_player_metrics(
                season=season,
                season_type=season_type,
                use_full_cache=use_full_cache,
            )
    except NbaApiError as exc:
        st.error(str(exc))
        leaderboard = pd.DataFrame()
    except Exception as exc:
        st.error(f"Failed to build season leaderboard: {exc}")
        leaderboard = pd.DataFrame()

    if leaderboard.empty:
        st.info("No season leaderboard data available yet for this selection.")
    else:
        leaderboard = leaderboard[leaderboard["Attempts"] >= int(min_attempts)].copy()
        ascending = sort_order == "Ascending"
        leaderboard = leaderboard.sort_values(by=sort_by, ascending=ascending)
        leaderboard = leaderboard.head(int(top_n)).copy()
        leaderboard = leaderboard.round({"FG%": 1, "xFG%": 1, "SMOE": 1, "Shot Diet Difficulty": 1})
        leaderboard = leaderboard.rename(columns={"player_name": "Player"})
        st.dataframe(
            leaderboard[
                ["Player", "Attempts", "FG%", "xFG%", "SMOE", "Shot Diet Difficulty"]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.subheader("Chart Guide")
g1, g2, g3 = st.columns(3)
g1.info("**Shot Volume**: Where this player shoots most often.")
g2.info("**Expected Make Chance**: How makeable shots are by location.")
g3.info("**Over/Under Expected**: Where results are above or below expectation.")
view_style = st.radio(
    "Shot Map Style",
    options=["Hexbin", "Court Zones", "Both"],
    horizontal=True,
)

tab_freq, tab_quality, tab_smoe = st.tabs(
    ["Shot Volume", "Expected Make Chance", "Over/Under Expected"]
)

with tab_freq:
    _render_shot_map_views(
        shots=shots,
        view_style=view_style,
        hexbin_fn=plot_frequency_heatmap,
        partitioned_fn=plot_frequency_partitioned_heatmap,
    )

with tab_quality:
    if not has_predictions:
        st.info("Train/load a model for this season to view expected make chance maps.")
    else:
        _render_shot_map_views(
            shots=shots,
            view_style=view_style,
            hexbin_fn=plot_quality_heatmap,
            partitioned_fn=plot_quality_partitioned_heatmap,
        )

with tab_smoe:
    if not has_predictions:
        st.info("Train/load a model for this season to view over/under expected maps.")
    else:
        _render_shot_map_views(
            shots=shots,
            view_style=view_style,
            hexbin_fn=plot_smoe_heatmap,
            partitioned_fn=plot_smoe_partitioned_heatmap,
        )

st.subheader("Best and Worst Spots")
st.caption(
    "These are grouped court areas with at least 15 shots. Positive SMOE means better-than-expected results."
)
if not has_predictions:
    st.info("Train/load a model for this season to generate best/worst area tables.")
else:
    spots_table = _build_location_table(shots, min_attempts=15)
    if spots_table.empty:
        st.info("Not enough attempts in any location bucket to summarize (minimum 15 attempts).")
    else:
        st.dataframe(spots_table, use_container_width=True, hide_index=True)

st.subheader("Model Diagnostics")
st.caption(
    "These checks tell you how reliable the expected-make model is. Lower numbers are better."
)
mdesc1, mdesc2 = st.columns(2)
mdesc1.info("**XGB Log Loss**: Penalizes very wrong confident predictions. Lower means better probability quality.")
mdesc2.info("**XGB Brier**: Average squared error of predicted probabilities. Lower means better calibration.")
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
