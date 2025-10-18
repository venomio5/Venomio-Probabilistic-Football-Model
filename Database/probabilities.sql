WITH minute_scores AS (
    SELECT
        sim_id,
        home_goals,
        away_goals,
        minute,
        ROW_NUMBER() OVER (PARTITION BY sim_id ORDER BY minute DESC) as rn
    FROM vpfm.simulation
    WHERE match_id = 4640
), final_scores AS (
    SELECT
        sim_id,
        home_goals,
        away_goals
    FROM minute_scores
    WHERE rn = 1
)
SELECT 
    COUNT(*) AS total_simulations,
    ROUND(100.0 * SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) / COUNT(*), 2) AS home_win_percent,
    ROUND(100.0 * SUM(CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END) / COUNT(*), 2) AS away_win_percent,
    ROUND(100.0 * SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) / COUNT(*), 2) AS draw_percent
FROM final_scores;