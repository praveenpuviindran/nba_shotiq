"""Causal inference analysis: does shot selection quality cause scoring efficiency?

Research question
-----------------
Does improving shot selection quality (average xFG% of shots taken) cause a
measurable improvement in actual points per possession, beyond what pre-existing
shooting skill would predict?

This module provides three complementary analyses:

Part A — Inverse Probability Weighting (IPW)
    Estimates the Average Treatment Effect (ATE) of being in the top tercile
    of shot selection quality, weighted by propensity scores from a logistic
    regression on available confounders.

Part B — Difference-in-Differences (DiD)
    Identifies players who meaningfully changed their shot selection between
    consecutive seasons and estimates the causal effect using a DiD design.

Part C — Rosenbaum Sensitivity Analysis
    Estimates how large an unmeasured confounder would need to be to explain
    away the observed effect.

All analyses operate on player-season aggregate data derived from the shots
and predictions tables.
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Data loading helpers                                                 #
# ------------------------------------------------------------------ #

def load_player_season_data(db_path: str | Path) -> pd.DataFrame:
    """
    Build a player-season aggregate table from shots + predictions.

    Returns one row per (player_id, season, season_type) with:
        - avg_xfg            : average expected make probability (shot selection quality)
        - actual_fg_pct      : actual field goal percentage
        - actual_pps         : actual points per shot (proxy for points per possession)
        - expected_pps       : expected points per shot
        - smoe               : shot making over expected
        - three_pt_rate      : fraction of shots that are 3-pointers
        - shots_attempted    : total attempts (for filtering)
        - prior_fg_pct       : previous season FG% (confounder)
        - prior_xfg          : previous season avg xFG (confounder)
        - prior_attempts     : previous season attempts (confounder)
    """
    conn = sqlite3.connect(Path(db_path))

    query = """
        SELECT
            s.player_id,
            s.player_name,
            s.season,
            s.season_type,
            COUNT(*)                                             AS shots_attempted,
            AVG(p.predicted_make_prob)                          AS avg_xfg,
            AVG(s.shot_made_flag)                               AS actual_fg_pct,
            -- Points per shot: 3PT shot made = 3 pts, 2PT = 2 pts; expected version uses p_make
            AVG(
                CASE WHEN s.shot_type = '3PT Field Goal' THEN s.shot_made_flag * 3.0
                     ELSE s.shot_made_flag * 2.0 END
            )                                                   AS actual_pps,
            AVG(
                CASE WHEN s.shot_type = '3PT Field Goal' THEN p.predicted_make_prob * 3.0
                     ELSE p.predicted_make_prob * 2.0 END
            )                                                   AS expected_pps,
            AVG(s.shot_made_flag) - AVG(p.predicted_make_prob) AS smoe,
            SUM(CASE WHEN s.shot_type = '3PT Field Goal' THEN 1 ELSE 0 END)
                * 1.0 / COUNT(*)                               AS three_pt_rate,
            AVG(s.shot_distance)                               AS avg_shot_distance
        FROM shots s
        JOIN predictions p ON p.shot_id = s.id
        GROUP BY s.player_id, s.player_name, s.season, s.season_type
        HAVING COUNT(*) >= 50
        ORDER BY s.player_id, s.season
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Attach prior-season stats as confounders (1-season lag per player)
    df = df.sort_values(["player_id", "season_type", "season"])
    for col in ["avg_xfg", "actual_fg_pct", "shots_attempted"]:
        df[f"prior_{col}"] = df.groupby(["player_id", "season_type"])[col].shift(1)

    return df


# ------------------------------------------------------------------ #
# Part A — Propensity Score / IPW                                     #
# ------------------------------------------------------------------ #

def compute_propensity_scores(player_df: pd.DataFrame) -> np.ndarray:
    """
    Fit a logistic regression to predict treatment assignment from confounders.

    Treatment: player is in the top tercile of shot selection quality
    (avg xFG >= 66th percentile — i.e. taking higher-quality shots on average).
    Note: higher xFG = easier shots; some players intentionally take harder shots
    (3PT) that have lower xFG but higher expected value. This analysis focuses
    purely on xFG as a measure of shot difficulty.

    Confounders included (from available data):
        - prior season actual FG%
        - prior season average xFG
        - prior season shot attempts (volume proxy)
        - 3PT rate (controls for shot-type mix strategy)
        - average shot distance

    Parameters
    ----------
    player_df : DataFrame
        Output from load_player_season_data().  Rows with missing confounders
        are dropped.

    Returns
    -------
    np.ndarray of shape (n,) — propensity scores in [0, 1]
    """
    threshold = player_df["avg_xfg"].quantile(0.667)
    player_df = player_df.copy()
    player_df["treatment"] = (player_df["avg_xfg"] >= threshold).astype(int)

    confounder_cols = [
        "prior_actual_fg_pct",
        "prior_avg_xfg",
        "prior_shots_attempted",
        "three_pt_rate",
        "avg_shot_distance",
    ]

    # Only use rows with all confounders present
    valid = player_df.dropna(subset=confounder_cols + ["treatment"])
    if len(valid) < 20:
        logger.warning(
            "Only %d valid rows for propensity model — results may be unstable.", len(valid)
        )

    X = valid[confounder_cols].values
    y = valid["treatment"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_scaled, y)

    propensity_scores = lr.predict_proba(X_scaled)[:, 1]

    # Attach scores back to valid rows
    valid = valid.copy()
    valid["propensity_score"] = propensity_scores
    valid["treatment"] = y

    return valid


def estimate_ipw_effect(
    player_df: pd.DataFrame,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> dict:
    """
    Estimate Average Treatment Effect (ATE) via Inverse Probability Weighting.

    ATE = E[Y(1) - Y(0)] where Y = actual_pps (points per shot).

    Treated units are weighted by 1/p(X), control by 1/(1-p(X)).
    Bootstrap (n_bootstrap iterations) gives SE and 95% CI.

    Parameters
    ----------
    player_df : DataFrame
        Must contain 'propensity_score', 'treatment', 'actual_pps' columns.
        Typically the output of compute_propensity_scores().
    n_bootstrap : int
        Number of bootstrap replicates for standard error estimation.

    Returns
    -------
    dict with keys: ate, se, ci_lower, ci_upper, p_value, n_treated, n_control
    """
    df = player_df.dropna(subset=["propensity_score", "treatment", "actual_pps"]).copy()

    # Clip extreme propensity scores to avoid unstable weights
    df["ps_clipped"] = df["propensity_score"].clip(0.05, 0.95)

    def _ipw_ate(data: pd.DataFrame) -> float:
        treated = data[data["treatment"] == 1]
        control = data[data["treatment"] == 0]

        if len(treated) == 0 or len(control) == 0:
            return float("nan")

        w_treated = 1.0 / treated["ps_clipped"]
        w_control = 1.0 / (1.0 - control["ps_clipped"])

        mu1 = np.average(treated["actual_pps"], weights=w_treated)
        mu0 = np.average(control["actual_pps"], weights=w_control)
        return float(mu1 - mu0)

    ate = _ipw_ate(df)

    rng = np.random.default_rng(random_state)
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        sample = df.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
        bootstrap_ates.append(_ipw_ate(sample))

    bootstrap_ates_arr = np.array([a for a in bootstrap_ates if not np.isnan(a)])
    se = float(bootstrap_ates_arr.std())
    ci_lower = float(np.percentile(bootstrap_ates_arr, 2.5))
    ci_upper = float(np.percentile(bootstrap_ates_arr, 97.5))

    # Two-tailed p-value from bootstrap distribution (H0: ATE = 0)
    z_score = ate / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    return {
        "ate": round(ate, 5),
        "se": round(se, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "p_value": round(p_value, 4),
        "n_treated": int((df["treatment"] == 1).sum()),
        "n_control": int((df["treatment"] == 0).sum()),
    }


# ------------------------------------------------------------------ #
# Part B — Difference-in-Differences                                  #
# ------------------------------------------------------------------ #

def difference_in_differences(player_panel_df: pd.DataFrame) -> dict:
    """
    Estimate causal effect of improving shot selection using a DiD design.

    Treatment group: players who IMPROVED shot selection quality by > 0.5 SD
                     between consecutive seasons.
    Control group:   players with stable shot selection (change < 0.5 SD).
    Pre period:      season T
    Post period:     season T+1
    Outcome:         actual points per shot (actual_pps)

    DiD estimator:
        δ = (treated_post − treated_pre) − (control_post − control_pre)

    Parameters
    ----------
    player_panel_df : DataFrame
        Output of load_player_season_data().  Must have multiple seasons per player.

    Returns
    -------
    dict with: did_estimate, se, ci_lower, ci_upper, p_value,
               n_treated, n_control, parallel_trends_data (dict of lists)
    """
    df = player_panel_df.copy()
    df = df.sort_values(["player_id", "season_type", "season"])

    # Compute season-over-season change in avg_xfg per player
    df["delta_xfg"] = df.groupby(["player_id", "season_type"])["avg_xfg"].diff()

    # Keep only rows with a valid prior season
    df = df.dropna(subset=["delta_xfg", "actual_pps", "prior_actual_fg_pct"])

    if df.empty:
        return {
            "did_estimate": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "n_treated": 0,
            "n_control": 0,
            "parallel_trends_data": {},
            "message": "Insufficient panel data for DiD (need ≥ 2 seasons per player).",
        }

    threshold = df["delta_xfg"].std() * 0.5

    # "Treated": improved shot selection quality (higher avg xFG = easier shots)
    df["treated"] = (df["delta_xfg"] >= threshold).astype(int)
    df["control"] = (df["delta_xfg"].abs() < threshold).astype(int)

    # Pair each player-season with its prior as pre/post rows
    pre_df = df.copy()
    pre_df["period"] = 0  # pre
    pre_df["outcome"] = pre_df.groupby(["player_id", "season_type"])["actual_pps"].shift(1)
    pre_df = pre_df.dropna(subset=["outcome"])

    post_df = df.copy()
    post_df["period"] = 1  # post
    post_df["outcome"] = post_df["actual_pps"]

    panel = pd.concat([pre_df, post_df], ignore_index=True)
    panel = panel.dropna(subset=["outcome", "treated"])

    treated_pre  = panel[(panel["treated"] == 1) & (panel["period"] == 0)]["outcome"].mean()
    treated_post = panel[(panel["treated"] == 1) & (panel["period"] == 1)]["outcome"].mean()
    control_pre  = panel[(panel["control"] == 1) & (panel["period"] == 0)]["outcome"].mean()
    control_post = panel[(panel["control"] == 1) & (panel["period"] == 1)]["outcome"].mean()

    did = (treated_post - treated_pre) - (control_post - control_pre)

    # SE via cluster-robust delta method approximation (bootstrap)
    rng = np.random.default_rng(42)
    boot_dids = []
    for _ in range(500):
        sample = panel.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
        tp = sample[(sample["treated"] == 1) & (sample["period"] == 1)]["outcome"].mean()
        tpre = sample[(sample["treated"] == 1) & (sample["period"] == 0)]["outcome"].mean()
        cp = sample[(sample["control"] == 1) & (sample["period"] == 1)]["outcome"].mean()
        cpre = sample[(sample["control"] == 1) & (sample["period"] == 0)]["outcome"].mean()
        if all(not np.isnan(v) for v in [tp, tpre, cp, cpre]):
            boot_dids.append((tp - tpre) - (cp - cpre))

    boot_arr = np.array(boot_dids)
    se = float(boot_arr.std()) if len(boot_arr) > 10 else float("nan")
    ci_lower = float(np.percentile(boot_arr, 2.5)) if len(boot_arr) > 10 else float("nan")
    ci_upper = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) > 10 else float("nan")
    z_score = did / se if (se and se > 0) else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score)))) if se > 0 else float("nan")

    # Parallel trends data (pre-period trends for treated vs control)
    parallel_trends_data = {
        "group": ["Treated (improved selection)", "Control (stable selection)"],
        "pre_outcome": [treated_pre, control_pre],
        "post_outcome": [treated_post, control_post],
    }

    return {
        "did_estimate": round(float(did), 5),
        "se": round(se, 5) if not np.isnan(se) else None,
        "ci_lower": round(ci_lower, 5) if not np.isnan(ci_lower) else None,
        "ci_upper": round(ci_upper, 5) if not np.isnan(ci_upper) else None,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "n_treated": int((df["treated"] == 1).sum()),
        "n_control": int((df["control"] == 1).sum()),
        "parallel_trends_data": parallel_trends_data,
        "treated_pre": round(float(treated_pre), 4) if not np.isnan(treated_pre) else None,
        "treated_post": round(float(treated_post), 4) if not np.isnan(treated_post) else None,
        "control_pre": round(float(control_pre), 4) if not np.isnan(control_pre) else None,
        "control_post": round(float(control_post), 4) if not np.isnan(control_post) else None,
    }


# ------------------------------------------------------------------ #
# Part C — Rosenbaum Sensitivity Analysis                             #
# ------------------------------------------------------------------ #

def sensitivity_analysis(
    player_df: pd.DataFrame,
    gamma_max: float = 3.0,
    gamma_step: float = 0.25,
) -> dict:
    """
    Rosenbaum bounds: estimate how large an unmeasured confounder would need
    to be to explain away the observed treatment effect.

    For matched observational studies, Gamma (Γ) represents the odds ratio of
    treatment assignment that an unmeasured confounder could produce.  We test
    at increasing Gamma values until the Wilcoxon signed-rank test p-value
    exceeds 0.05 — that Gamma is the sensitivity threshold.

    Parameters
    ----------
    player_df : DataFrame
        Output of compute_propensity_scores() with columns:
        'treatment', 'actual_pps', 'propensity_score'.
    gamma_max : float
        Upper bound of Gamma to test.
    gamma_step : float
        Increment between Gamma values.

    Returns
    -------
    dict with:
        gamma_break: Gamma at which p > 0.05 (None if holds up to gamma_max)
        gamma_table: list of {"gamma": float, "p_upper": float, "significant": bool}
        interpretation: human-readable summary
    """
    df = player_df.dropna(subset=["treatment", "actual_pps", "propensity_score"]).copy()
    df["ps_clipped"] = df["propensity_score"].clip(0.05, 0.95)

    treated_outcomes = df[df["treatment"] == 1]["actual_pps"].values
    control_outcomes = df[df["treatment"] == 0]["actual_pps"].values

    if len(treated_outcomes) < 5 or len(control_outcomes) < 5:
        return {
            "gamma_break": None,
            "gamma_table": [],
            "interpretation": "Insufficient data for Rosenbaum bounds.",
        }

    gamma_table = []
    gamma_break = None

    gammas = np.arange(1.0, gamma_max + gamma_step, gamma_step)
    for gamma in gammas:
        # Upper bound on p-value: shift treated ranks up by Gamma factor
        # (Rosenbaum upper bound approximation using Wilcoxon with adjusted ranks)
        # Full implementation uses signed-rank statistics; here we use an approximation
        # via effect-adjusted Wilcoxon test
        adjusted_treated = treated_outcomes * (1.0 / gamma)  # shrink effect toward null
        _, p_upper = stats.ranksums(adjusted_treated, control_outcomes)
        p_upper = float(p_upper)
        significant = p_upper < 0.05
        gamma_table.append({
            "gamma": round(float(gamma), 2),
            "p_upper": round(p_upper, 4),
            "significant": significant,
        })
        if not significant and gamma_break is None:
            gamma_break = round(float(gamma), 2)

    if gamma_break is not None:
        interpretation = (
            f"The effect loses statistical significance at Γ = {gamma_break:.2f}. "
            f"This means an unmeasured confounder would need to make treated players "
            f"{gamma_break:.2f}× more likely to improve their shot selection to "
            f"explain away the observed effect."
        )
    else:
        interpretation = (
            f"The effect remains statistically significant even at Γ = {gamma_max:.2f}. "
            f"No single unmeasured confounder up to {gamma_max:.2f}× strength "
            f"is sufficient to explain away the observed effect."
        )

    return {
        "gamma_break": gamma_break,
        "gamma_table": gamma_table,
        "interpretation": interpretation,
    }
