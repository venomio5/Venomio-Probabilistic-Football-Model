import core
import ast
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import numpy as np

"""
Before processing new data, update the pre defined RAS for old matches.
"""
non_pdras_matches_df = core.DB.select("""
    SELECT 
        md.detail_id,
        md.match_id,
        mi.league_id,
        md.teamA_players,
        md.teamB_players,
        md.minutes_played
    FROM match_detail md
    JOIN match_info mi ON mi.match_id = md.match_id
""")

non_pdras_matches_df['teamA_players'] = non_pdras_matches_df['teamA_players'].apply(
    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
)
non_pdras_matches_df['teamB_players'] = non_pdras_matches_df['teamB_players'].apply(
    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
)

players_needed = set()
for _, row in non_pdras_matches_df.iterrows():
    players_needed.update(row['teamA_players'])
    players_needed.update(row['teamB_players'])

if players_needed:
    placeholders = ','.join(['%s'] * len(players_needed))
    players_sql = (
        f"SELECT player_id, off_sh_coef, def_sh_coef "
        f"FROM players_data "
        f"WHERE player_id IN ({placeholders});"
    )
    players_coef_df = core.DB.select(players_sql, list(players_needed))
    off_sh_coef_dict = players_coef_df.set_index("player_id")["off_sh_coef"].to_dict()
    def_sh_coef_dict = players_coef_df.set_index("player_id")["def_sh_coef"].to_dict()
else:
    off_sh_coef_dict, def_sh_coef_dict = {}, {}

baseline_df = core.DB.select("SELECT league_id, sh_baseline_coef FROM league_data;")
baseline_dict = baseline_df.set_index("league_id")["sh_baseline_coef"].fillna(0).to_dict()

for _, row in non_pdras_matches_df.iterrows():
    minutes = row['minutes_played']
    league_id = row['league_id']
    baseline = baseline_dict.get(league_id, 0.0)

    teamA_ids = row['teamA_players']
    teamB_ids = row['teamB_players']

    teamA_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamA_ids)
    teamB_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamB_ids)
    teamA_pdras = (baseline + teamA_offense - teamB_defense) * minutes

    teamB_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamB_ids)
    teamA_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamA_ids)
    teamB_pdras = (baseline + teamB_offense - teamA_defense) * minutes

    core.DB.execute(
        "UPDATE match_detail SET teamA_pdras = %s, teamB_pdras = %s WHERE detail_id = %s",
        (teamA_pdras, teamB_pdras, row['detail_id'])
    )
