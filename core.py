import databasemanager
import numpy as np
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import json

# ------------------------------ Fetch & Save data ------------------------------
# ------------------------------ Process data ------------------------------
class process_data:
    def __init__(self):
        """
        Class to reset the players_data table and fill it with new data.
        """
        self.db = databasemanager.DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")
        self.db.execute("TRUNCATE TABLE players_data;")

        self.insert_players_basics()

        self.update_players_sh_coef()


    def insert_players_basics(self):
        """
        Function to insert basic information from all players into players_data from match detail without duplicating.
        """
        sql = """
        SELECT md.teamA_players, md.teamB_players, mi.match_home_team_id, mi.match_away_team_id 
        FROM match_detail md 
        JOIN match_info mi ON md.match_id = mi.match_id 
        """
        result = self.db.select(sql, ())
        
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
        inserted = self.db.execute(insert_sql, players_to_insert, many=True)
        
        return inserted

    def update_players_sh_coef(self):
        """
        Function to update players coefficients per league
        """
        def extract_player_ids(players_json_str):
            players = json.loads(players_json_str)
            return [player['id'] for player in players]
        
        league_id_df = self.db.select("SELECT league_id FROM league_data")

        for league_id in league_id_df['league_id'].tolist():
            league_matches_df =self.db.select(f"SELECT match_id FROM match_info WHERE match_league_id = {league_id}")
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
            WHERE match_id IN ({matches_ids_placeholder}) 
            """
            matches_details_df = self.db.select(matches_sql, matches_ids)
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
                self.db.execute(update_coef_query, (off_sh, def_sh, player))

# ------------------------------ Monte Carlo ------------------------------
# ------------------------------ Trading ------------------------------
# Init
process_data()