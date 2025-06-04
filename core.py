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
        self.update_players_sh_coef()
        self.update_players_st_coef()

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

        players_to_insert = []
        for _, row in result.iterrows():
            teamA_players = json.loads(row["teamA_players"])
            teamB_players = json.loads(row["teamB_players"])
            home_team = int(row["match_home_team_id"])
            away_team = int(row["match_away_team_id"])
        
            for player in teamA_players:
                players_to_insert.append((player["id"], player["name"], home_team))
            
            for player in teamB_players:
                players_to_insert.append((player["id"], player["name"], away_team))
        
        insert_sql = "INSERT IGNORE INTO players_data (player_id, player_name, current_team) VALUES (%s, %s, %s)"
        inserted = DB.execute(insert_sql, players_to_insert, many=True)
        
        return inserted

    def update_players_sh_coef(self):
        """
        Function to update players shots coefficients per league
        """
        
        league_id_df = DB.select("SELECT league_id FROM league_data")

        for league_id in league_id_df['league_id'].tolist():
            league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE match_league_id = {league_id}")
            matches_ids = league_matches_df['match_id'].tolist()
            matches_ids_placeholder = ','.join(['%s'] * len(matches_ids))
            matches_sql = f"""
            SELECT 
                teamA_players, 
                teamB_players, 
                (teamA_headers + teamA_footers) AS teamA_shots, 
                (teamB_headers + teamB_footers) AS teamB_shots, 
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
                teamA_shots = row['teamA_shots']
                teamB_shots = row['teamB_shots']

                for p in teamA_players:
                    rows.append(row_num)
                    cols.append(players_to_index[p])
                    data_vals.append(1)
                for p in teamB_players:
                    rows.append(row_num)
                    cols.append(num_players + players_to_index[p])
                    data_vals.append(-1)
                y.append(teamA_shots / minutes)
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
                y.append(teamB_shots / minutes)
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
                update_coef_query = """
                UPDATE players_data
                SET off_sh_coef = %s, def_sh_coef = %s
                WHERE player_id = %s
                """
                DB.execute(update_coef_query, (off_sh, def_sh, player))

    def update_players_st_coef(self):
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
# ------------------------------ Monte Carlo ------------------------------
"""
Buidl contextual xgboost when predicting
"""
# ------------------------------ Trading ------------------------------
# Init
extract_data()
process_data()