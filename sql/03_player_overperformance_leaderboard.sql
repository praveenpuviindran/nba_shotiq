-- =============================================================================
-- Query 03: Player Overperformance Leaderboard (SMOE with Statistical Filtering)
-- =============================================================================
-- Purpose:
--   Rank players by Shot Making Over Expected (SMOE) and filter out those whose
--   overperformance may be statistical noise.  Only players whose lower 95%
--   confidence interval bound is > 0 are included — this means the observed
--   SMOE is unlikely to be purely chance.
--
-- Technique:
--   CTEs, manual STDDEV (SQLite-compatible formula),
--   95% CI half-width, NTILE, RANK.
--
-- SMOE definition:
--   SMOE = actual FG% − expected FG% (xFG%)
--   Positive SMOE → player makes more than model expects.
--   Negative SMOE → player underperforms shot quality.
--
-- Tables used:
--   shots       — shot metadata, outcomes, player names
--   predictions — model expected make probabilities
--
-- Usage:
--   sqlite3 data/nba_shotiq.db < sql/03_player_overperformance_leaderboard.sql
-- =============================================================================

WITH player_stats AS (
    SELECT
        s.player_id,
        s.player_name,
        s.season,
        s.season_type,
        COUNT(*)                                          AS total_shots,
        ROUND(AVG(p.predicted_make_prob), 4)              AS avg_xfg,
        ROUND(AVG(s.shot_made_flag), 4)                   AS avg_actual_fg,
        ROUND(AVG(s.shot_made_flag) - AVG(p.predicted_make_prob), 4) AS smoe,
        -- SQLite manual STDDEV: sqrt(E[x²] - E[x]²)
        ROUND(
            SQRT(
                AVG((s.shot_made_flag - p.predicted_make_prob) *
                    (s.shot_made_flag - p.predicted_make_prob))
                - AVG(s.shot_made_flag - p.predicted_make_prob)
                  * AVG(s.shot_made_flag - p.predicted_make_prob)
            ),
            6
        )                                                 AS smoe_std,
        -- 95% CI half-width = 1.96 × std / sqrt(n)
        ROUND(
            1.96 * SQRT(
                AVG((s.shot_made_flag - p.predicted_make_prob) *
                    (s.shot_made_flag - p.predicted_make_prob))
                - AVG(s.shot_made_flag - p.predicted_make_prob)
                  * AVG(s.shot_made_flag - p.predicted_make_prob)
            ) / SQRT(COUNT(*)),
            6
        )                                                 AS smoe_ci_halfwidth
    FROM shots s
    JOIN predictions p ON p.shot_id = s.id
    GROUP BY s.player_id, s.player_name, s.season, s.season_type
    HAVING COUNT(*) >= 100
)
SELECT
    player_id,
    player_name,
    season,
    season_type,
    total_shots,
    avg_xfg,
    avg_actual_fg,
    smoe,
    smoe_std,
    smoe_ci_halfwidth,
    ROUND(smoe - smoe_ci_halfwidth, 4)  AS smoe_ci_lower,
    ROUND(smoe + smoe_ci_halfwidth, 4)  AS smoe_ci_upper,
    NTILE(5) OVER (ORDER BY smoe DESC)  AS smoe_quintile,
    RANK()   OVER (ORDER BY smoe DESC)  AS smoe_rank
FROM player_stats
WHERE smoe - smoe_ci_halfwidth > 0      -- statistically significant overperformers only
ORDER BY smoe_rank;
