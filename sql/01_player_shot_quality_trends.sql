-- =============================================================================
-- Query 01: Player Shot Quality Trends (Month-Over-Month)
-- =============================================================================
-- Purpose:
--   Track how each player's shot selection quality changes month-over-month.
--   Shot quality is measured by average expected make probability (xFG%) from
--   the predictions table and actual FG% from shot outcomes.
--
-- Technique:
--   Window functions — LAG (previous month xFG), AVG OVER (3-month rolling).
--
-- Tables used:
--   shots       — shot metadata and outcomes
--   predictions — model-generated expected make probabilities (p_make)
--
-- Usage:
--   sqlite3 data/nba_shotiq.db < sql/01_player_shot_quality_trends.sql
-- =============================================================================

WITH monthly_metrics AS (
    SELECT
        s.player_id,
        s.player_name,
        s.season,
        s.season_type,
        strftime('%Y-%m', s.game_date)            AS month,
        AVG(p.predicted_make_prob)                AS avg_xfg,
        AVG(s.shot_made_flag)                     AS actual_fg,
        COUNT(*)                                  AS shots_attempted,
        SUM(
            CASE WHEN s.shot_zone_basic = 'Above the Break 3'
                      OR s.shot_zone_basic = 'Backcourt'
                      OR s.shot_zone_basic = 'Left Corner 3'
                      OR s.shot_zone_basic = 'Right Corner 3'
                 THEN 1 ELSE 0 END
        ) * 1.0 / COUNT(*)                        AS three_pt_rate,
        AVG(s.shot_distance)                      AS avg_shot_distance
    FROM shots s
    JOIN predictions p ON p.shot_id = s.id
    WHERE s.game_date IS NOT NULL
    GROUP BY s.player_id, s.player_name, s.season, s.season_type, month
),
with_trends AS (
    SELECT
        *,
        LAG(avg_xfg, 1) OVER (
            PARTITION BY player_id, season, season_type
            ORDER BY month
        )                                         AS prev_month_xfg,
        avg_xfg - LAG(avg_xfg, 1) OVER (
            PARTITION BY player_id, season, season_type
            ORDER BY month
        )                                         AS xfg_delta,
        AVG(avg_xfg) OVER (
            PARTITION BY player_id, season, season_type
            ORDER BY month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )                                         AS rolling_3mo_xfg,
        LAG(actual_fg, 1) OVER (
            PARTITION BY player_id, season, season_type
            ORDER BY month
        )                                         AS prev_month_fg,
        actual_fg - LAG(actual_fg, 1) OVER (
            PARTITION BY player_id, season, season_type
            ORDER BY month
        )                                         AS fg_delta
    FROM monthly_metrics
)
SELECT
    player_id,
    player_name,
    season,
    season_type,
    month,
    ROUND(avg_xfg, 4)          AS avg_xfg_pct,
    ROUND(actual_fg, 4)        AS actual_fg_pct,
    ROUND(actual_fg - avg_xfg, 4) AS smoe,
    ROUND(prev_month_xfg, 4)   AS prev_month_xfg_pct,
    ROUND(xfg_delta, 4)        AS xfg_delta,
    ROUND(rolling_3mo_xfg, 4)  AS rolling_3mo_avg_xfg,
    ROUND(three_pt_rate, 4)    AS three_pt_rate,
    ROUND(avg_shot_distance, 2) AS avg_shot_distance,
    shots_attempted
FROM with_trends
WHERE shots_attempted >= 20
ORDER BY player_name, season, month;
