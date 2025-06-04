import databasemanager
import numpy as np
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import json

DB = databasemanager.DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")

# --------------- Useful functions ---------------
def extract_player_ids(players_json_str):
    players = json.loads(players_json_str)
    return [player['id'] for player in players]

# ------------------------------ Fetch & Remove Data ------------------------------
class extract_data:
    def __init__(self):
        """"""
        self.update_pdras()

    def update_pdras(self):
        """
        Before fetching new data, update the pre defined RAS from old matches.
        """
        non_pdras_matches_df = DB.select("SELECT match_id, teamA_players, teamB_players, minutes_played FROM match_detail WHERE teamA_pdras IS NULL OR teamB_pdras IS NULL;")
        non_pdras_matches_df['teamA_player_ids'] = non_pdras_matches_df['teamA_players'].apply(extract_player_ids)
        non_pdras_matches_df['teamB_player_ids'] = non_pdras_matches_df['teamB_players'].apply(extract_player_ids)
        cols_to_drop = ['teamA_players', 'teamB_players']
        non_pdras_matches_df = non_pdras_matches_df.drop(columns=cols_to_drop)

        players_needed = set()
        for idx, row in non_pdras_matches_df.iterrows():
            players_needed.update(row['teamA_player_ids'])
            players_needed.update(row['teamB_player_ids'])

        if players_needed:
            players_str_placeholder = ','.join(['%s'] * len(players_needed))
            players_sql = f"SELECT player_id, off_sh_coef, def_sh_coef FROM players_data WHERE player_id IN ({players_str_placeholder});"
            players_coef_df = DB.select(players_sql, list(players_needed))
            off_sh_coef_dict = players_coef_df.set_index("player_id")["off_sh_coef"].to_dict()
            def_sh_coef_dict = players_coef_df.set_index("player_id")["def_sh_coef"].to_dict()
        else:
            off_sh_coef_dict = {}
            def_sh_coef_dict = {}

        for idx, row in non_pdras_matches_df.iterrows():
            minutes = row['minutes_played']
            teamA_ids = row['teamA_player_ids']
            teamB_ids = row['teamB_player_ids']

            teamA_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_pdras = (teamA_offense - teamB_defense) * minutes

            teamB_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_pdras = (teamB_offense - teamA_defense) * minutes

            DB.execute("UPDATE match_detail SET teamA_pdras = %s, teamB_pdras = %s WHERE match_id = %s", (teamA_pdras, teamB_pdras, row['match_id']))
# ------------------------------ Process data ------------------------------
class process_data:
    def __init__(self):
        """
        Class to reset the players_data table and fill it with new data.
        """

        DB.execute("TRUNCATE TABLE players_data;")

        self.insert_players_basics()
        self.update_players_shots_coef()
        self.update_players_totals()
        self.update_players_xg_coef()

    def insert_players_basics(self):
        """
        Function to insert basic information from all players into players_data from match detail without duplicating.
        """
        sql = """
        SELECT md.teamA_players, md.teamB_players, mi.match_home_team_id, mi.match_away_team_id 
        FROM match_detail md 
        JOIN match_info mi ON md.match_id = mi.match_id 
        """
        result = DB.select(sql, ())
        
        if result.empty:
            return 0

        players_set = set()
        for _, row in result.iterrows():
            teamA_players = json.loads(row["teamA_players"])
            teamB_players = json.loads(row["teamB_players"])
            home_team = int(row["match_home_team_id"])
            away_team = int(row["match_away_team_id"])
        
            for player in teamA_players:
                players_set.add((player["id"], player["name"], home_team))
            
            for player in teamB_players:
                players_set.add((player["id"], player["name"], away_team))
        
        insert_sql = "INSERT IGNORE INTO players_data (player_id, player_name, current_team) VALUES (%s, %s, %s)"
        inserted = DB.execute(insert_sql, list(players_set), many=True)

    def update_players_shots_coef(self):
        """
        Function to update players shot types coefficients per league
        """
        league_id_df = DB.select("SELECT league_id FROM league_data")

        for league_id in league_id_df['league_id'].tolist():
            for shot_type in ["headers", "footers"]:
                league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE match_league_id = {league_id}")
                matches_ids = league_matches_df['match_id'].tolist()
                matches_ids_placeholder = ','.join(['%s'] * len(matches_ids))
                matches_sql = f"""
                SELECT 
                    teamA_players, 
                    teamB_players, 
                    teamA_{shot_type}, 
                    teamB_{shot_type}, 
                    minutes_played 
                FROM match_detail 
                WHERE match_id IN ({matches_ids_placeholder});
                """
                matches_details_df = DB.select(matches_sql, matches_ids)
                matches_details_df['teamA_player_ids'] = matches_details_df['teamA_players'].apply(extract_player_ids)
                matches_details_df['teamB_player_ids'] = matches_details_df['teamB_players'].apply(extract_player_ids)
                cols_to_drop = ['teamA_players', 'teamB_players']
                matches_details_df = matches_details_df.drop(columns=cols_to_drop)

                players_set = set()
                for idx, row in matches_details_df.iterrows():
                    players_set.update(row['teamA_player_ids'])
                    players_set.update(row['teamB_player_ids'])
                players = sorted(list(players_set))
                num_players = len(players)
                players_to_index = {player: idx for idx, player in enumerate(players)}

                rows = []
                cols = []
                data_vals = []
                y = []
                sample_weights = []
                row_num = 0

                for idx, row in matches_details_df.iterrows():
                    minutes = row['minutes_played']
                    if minutes == 0:
                        continue
                    teamA_players = row['teamA_player_ids']
                    teamB_players = row['teamB_player_ids']
                    teamA_st = row[f'teamA_{shot_type}']
                    teamB_st = row[f'teamB_{shot_type}']

                    for p in teamA_players:
                        rows.append(row_num)
                        cols.append(players_to_index[p])
                        data_vals.append(1)
                    for p in teamB_players:
                        rows.append(row_num)
                        cols.append(num_players + players_to_index[p])
                        data_vals.append(-1)
                    y.append(teamA_st / minutes)
                    sample_weights.append(minutes)
                    row_num += 1

                    for p in teamB_players:
                        rows.append(row_num)
                        cols.append(players_to_index[p])
                        data_vals.append(1)
                    for p in teamA_players:
                        rows.append(row_num)
                        cols.append(num_players + players_to_index[p])
                        data_vals.append(-1)
                    y.append(teamB_st / minutes)
                    sample_weights.append(minutes)
                    row_num += 1

                X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
                y_array = np.array(y)
                sample_weights_array = np.array(sample_weights)

                ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg')
                ridge.fit(X, y_array, sample_weight=sample_weights_array)

                offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
                defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))

                for player in players:
                    off_sh = offensive_ratings[player]
                    def_sh = defensive_ratings[player]
                    update_coef_query = f"""
                    UPDATE players_data
                    SET off_{shot_type}_coef = %s, def_{shot_type}_coef = %s
                    WHERE player_id = %s
                    """
                    DB.execute(update_coef_query, (off_sh, def_sh, player))
        sum_coef_sql = """
        UPDATE players_data
        SET off_sh_coef = COALESCE(off_headers_coef, 0) + COALESCE(off_footers_coef, 0),
            def_sh_coef = COALESCE(def_headers_coef, 0) + COALESCE(def_footers_coef, 0)
        """
        DB.execute(sum_coef_sql)

    def update_players_totals(self):
        """
        Function to sum all information from all players into players_data from match breakdown.
        """
        players_id_df = DB.select("SELECT DISTINCT player_id FROM players_data")
        
        for player_id in players_id_df["player_id"].tolist():
            agg_query = """
                SELECT 
                    COALESCE(SUM(headers / NULLIF(player_minutes_played, 0)), 0) AS headers,
                    COALESCE(SUM(footers / NULLIF(player_minutes_played, 0)), 0) AS footers,
                    COALESCE(SUM(key_passes / NULLIF(player_minutes_played, 0)), 0) AS key_passes,
                    COALESCE(SUM(non_assisted_footers / NULLIF(player_minutes_played, 0)), 0) AS non_assisted_footers
                FROM match_breakdown
                WHERE player_id = %s
            """
            agg_result = DB.select(agg_query, (player_id,))
            if agg_result.empty:
                continue

            row = agg_result.iloc[0]
            update_query = """
                UPDATE players_data
                SET 
                    headers = %s,
                    footers = %s,
                    key_passes = %s,
                    non_assisted_footers = %s
                WHERE player_id = %s
            """
            DB.execute(update_query, (
                row["headers"],
                row["footers"],
                row["key_passes"],
                row["non_assisted_footers"],
                player_id,
            ))

    def update_players_xg_coef(self):
        """
        Function to update players xg coefficients per league
        """
        league_id_df = DB.select("SELECT league_id FROM league_data")
        
        for league_id in league_id_df['league_id'].tolist():
            for shot_type in ["headers", "footers"]:
                prefix = "h" if shot_type == "headers" else "f"
                league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE match_league_id = {league_id}")
                matches_ids = league_matches_df['match_id'].tolist()
                if not matches_ids:
                    continue
                matches_ids_placeholder = ','.join(['%s'] * len(matches_ids))
                matches_sql = f"""
                SELECT 
                    teamA_players, 
                    teamB_players, 
                    teamA_{shot_type}, 
                    teamB_{shot_type},
                    teamA_{prefix}xg as teamA_xg,
                    teamB_{prefix}xg as teamB_xg
                FROM match_detail 
                WHERE match_id IN ({matches_ids_placeholder});
                """
                matches_details_df = DB.select(matches_sql, matches_ids)
                matches_details_df['teamA_player_ids'] = matches_details_df['teamA_players'].apply(extract_player_ids)
                matches_details_df['teamB_player_ids'] = matches_details_df['teamB_players'].apply(extract_player_ids)
                cols_to_drop = ['teamA_players', 'teamB_players']
                matches_details_df = matches_details_df.drop(columns=cols_to_drop)
                
                players_set = set()
                for idx, row in matches_details_df.iterrows():
                    players_set.update(row['teamA_player_ids'])
                    players_set.update(row['teamB_player_ids'])
                players = sorted(list(players_set))
                num_players = len(players)
                players_to_index = {player: idx for idx, player in enumerate(players)}
                
                rows = []
                cols = []
                data_vals = []
                y = []
                sample_weights = []
                row_num = 0
                
                for idx, row in matches_details_df.iterrows():
                    shots_teamA = row[f'teamA_{shot_type}']
                    shots_teamB = row[f'teamB_{shot_type}']
                    if shots_teamA > 0:
                        xg_teamA = row['teamA_xg']
                        for p in row['teamA_player_ids']:
                            rows.append(row_num)
                            cols.append(players_to_index[p])
                            data_vals.append(1)
                        for p in row['teamB_player_ids']:
                            rows.append(row_num)
                            cols.append(num_players + players_to_index[p])
                            data_vals.append(-1)
                        y.append(xg_teamA / shots_teamA)
                        sample_weights.append(shots_teamA)
                        row_num += 1
                    if shots_teamB > 0:
                        xg_teamB = row['teamB_xg']
                        for p in row['teamB_player_ids']:
                            rows.append(row_num)
                            cols.append(players_to_index[p])
                            data_vals.append(1)
                        for p in row['teamA_player_ids']:
                            rows.append(row_num)
                            cols.append(num_players + players_to_index[p])
                            data_vals.append(-1)
                        y.append(xg_teamB / shots_teamB)
                        sample_weights.append(shots_teamB)
                        row_num += 1
                
                if row_num == 0:
                    continue
                
                X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
                y_array = np.array(y)
                sample_weights_array = np.array(sample_weights)
                
                ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg')
                ridge.fit(X, y_array, sample_weight=sample_weights_array)
                
                offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
                defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))
                
                for player in players:
                    off_coef = offensive_ratings[player]
                    def_coef = defensive_ratings[player]
                    if shot_type == "headers":
                        update_coef_query = f"""
                        UPDATE players_data
                        SET off_hxg_coef = %s, def_hxg_coef = %s
                        WHERE player_id = %s
                        """
                    else:
                        update_coef_query = f"""
                        UPDATE players_data
                        SET off_fxg_coef = %s, def_fxg_coef = %s
                        WHERE player_id = %s
                        """
                    DB.execute(update_coef_query, (off_coef, def_coef, player))

# ------------------------------ Monte Carlo ------------------------------
"""
Build contextual xgboost when predicting
"""
# ------------------------------ Trading ------------------------------
# Init
extract_data()
process_data()