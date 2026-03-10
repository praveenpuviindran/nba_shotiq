-- =============================================================================
-- Query 04: Shot Selection by Season and Zone
-- =============================================================================
-- Purpose:
--   Analyze how shot selection patterns vary across seasons and shot zones.
--   Reveals trends in shot type mix (3PT vs paint vs mid-range) and how
--   each zone's efficiency has evolved over time.
--
-- Technique:
--   Multi-level GROUP BY, conditional aggregation, window AVG OVER season.
--
-- Note:
--   Quarter and score differential are not in the ShotIQ schema.
--   This query uses season as the time axis and shot_zone_basic as the
--   situational axis — both are always available.
--
-- Tables used:
--   shots       — shot metadata and outcomes
--   predictions — model expected make probabilities
--
-- Usage:
--   sqlite3 data/nba_shotiq.db < sql/04_game_flow_shot_selection.sql
-- =============================================================================

SELECT
    s.season,
    s.season_type,
    s.shot_zone_basic                                          AS shot_zone,
    COUNT(*)                                                   AS total_shots,
    ROUND(AVG(p.predicted_make_prob), 4)                       AS avg_xfg,
    ROUND(AVG(s.shot_made_flag), 4)                            AS actual_fg_pct,
    ROUND(AVG(s.shot_made_flag) - AVG(p.predicted_make_prob), 4) AS smoe,
    ROUND(AVG(s.shot_distance), 2)                             AS avg_distance,
    -- 3PT rate within this zone
    SUM(
        CASE WHEN s.shot_type = '3PT Field Goal' THEN 1 ELSE 0 END
    ) * 1.0 / COUNT(*)                                        AS three_pt_rate,
    -- Paint (restricted area or in-the-paint) shot rate
    SUM(
        CASE WHEN s.shot_zone_basic IN ('Restricted Area', 'In The Paint (Non-RA)')
             THEN 1 ELSE 0 END
    ) * 1.0 / COUNT(*)                                        AS paint_rate,
    -- Season-level rolling average xFG for this zone (window over seasons)
    ROUND(AVG(AVG(p.predicted_make_prob)) OVER (
        PARTITION BY s.shot_zone_basic, s.season_type
        ORDER BY s.season
        ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    ), 4)                                                     AS rolling_2season_xfg,
    -- Rank each zone by SMOE within this season
    RANK() OVER (
        PARTITION BY s.season, s.season_type
        ORDER BY AVG(s.shot_made_flag) - AVG(p.predicted_make_prob) DESC
    )                                                         AS smoe_rank_in_season
FROM shots s
JOIN predictions p ON p.shot_id = s.id
GROUP BY s.season, s.season_type, s.shot_zone_basic
HAVING COUNT(*) >= 50
ORDER BY s.season DESC, total_shots DESC;
