CREATE OR REPLACE VIEW VPFM.contextual_records AS
SELECT 
	mi.match_id,
	mi.home_team_id,
	mi.away_team_id,
	mi.home_elevation_dif,
	mi.away_elevation_dif,
	mi.away_travel,
	mi.temperature_c,
	mi.is_raining,
	mi.date,
	sum(md.teamA_pdras) AS teamA_pdras,
	sum(md.teamB_pdras) AS teamB_pdras,
	sum(md.minutes_played) AS minutes_played,
	md.match_state,
	md.player_dif,
	(sum(md.teamA_headers) + sum(md.teamA_footers)) AS home_shots,
	(sum(md.teamB_headers) + sum(md.teamB_footers)) AS away_shots
FROM vpfm.match_info mi
JOIN vpfm.match_detail md 
	ON mi.match_id = md.match_id
GROUP BY mi.match_id, md.match_state, md.player_dif