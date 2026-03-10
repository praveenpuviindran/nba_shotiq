-- =============================================================================
-- Query 02: Shot Context Cohort Analysis
-- =============================================================================
-- Purpose:
--   Compare shooting efficiency across situational cohorts defined by shot
--   distance and defensive context (approximated by shot zone).
--   Reveals which zone + distance combinations produce over- or under-performance.
--
-- Technique:
--   CASE WHEN cohort definition, RANK() OVER PARTITION BY zone.
--
-- Note:
--   Shot clock and defender distance are not in the ShotIQ schema.
--   Distance cohorts use shot_distance; zone uses shot_zone_basic.
--
-- Tables used:
--   shots       — shot metadata and outcomes
--   predictions — model-generated expected make probabilities
--
-- Usage:
--   sqlite3 data/nba_shotiq.db < sql/02_shot_context_cohort_analysis.sql
-- =============================================================================

SELECT
    CASE
        WHEN s.shot_distance <= 5  THEN 'at_rim'
        WHEN s.shot_distance <= 10 THEN 'short_mid'
        WHEN s.shot_distance <= 18 THEN 'mid_range'
        ELSE                            'long_range'
    END                                              AS distance_cohort,
    CASE
        WHEN s.shot_zone_area IN ('Center(C)')       THEN 'straight_on'
        WHEN s.shot_zone_area LIKE '%Left%'          THEN 'left_side'
        WHEN s.shot_zone_area LIKE '%Right%'         THEN 'right_side'
        ELSE                                              'other'
    END                                              AS angle_cohort,
    s.shot_zone_basic                                AS shot_zone,
    COUNT(*)                                         AS attempts,
    ROUND(AVG(s.shot_made_flag), 4)                  AS actual_fg_pct,
    ROUND(AVG(p.predicted_make_prob), 4)             AS avg_xfg_pct,
    ROUND(AVG(s.shot_made_flag) - AVG(p.predicted_make_prob), 4) AS smoe,
    RANK() OVER (
        PARTITION BY s.shot_zone_basic
        ORDER BY AVG(s.shot_made_flag) - AVG(p.predicted_make_prob) DESC
    )                                                AS smoe_rank_within_zone,
    NTILE(4) OVER (
        ORDER BY AVG(s.shot_made_flag) - AVG(p.predicted_make_prob) DESC
    )                                                AS smoe_quartile
FROM shots s
JOIN predictions p ON p.shot_id = s.id
GROUP BY distance_cohort, angle_cohort, shot_zone
HAVING COUNT(*) >= 30
ORDER BY shot_zone, smoe_rank_within_zone;
