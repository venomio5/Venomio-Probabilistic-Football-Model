import databasemanager
import json

# ------------------------------ Fetch & Save data ------------------------------
# --------------- fds ---------------
# ------------------------------ Process data ------------------------------
# --------------- Fill Players Data  ---------------
class fill_players_data:
    def __init__(self):
        """
        Reset the table and fill with new data
        """
        self.db = databasemanager.DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")
        self.db.execute("TRUNCATE TABLE players_data;")

        self.insert_players_basics()


    def insert_players_basics(self):
        sql = """
        SELECT md.teamA_players, md.teamB_players, mi.match_home_team_id, mi.match_away_team_id 
        FROM match_detail md 
        JOIN match_info mi ON md.match_id = mi.match_id 
        """
        result = self.db.select(sql, ())
        
        if result.empty:
            return 0

        row = result.iloc[0]
        teamA_players = json.loads(row["teamA_players"])
        teamB_players = json.loads(row["teamB_players"])
        home_team = int(row["match_home_team_id"])
        away_team = int(row["match_away_team_id"])

        players_to_insert = []
        
        for player in teamA_players:
            players_to_insert.append((player["id"], player["name"], home_team))
        
        for player in teamB_players:
            players_to_insert.append((player["id"], player["name"], away_team))
        
        insert_sql = "INSERT IGNORE INTO players_data (player_id, player_name, current_team) VALUES (%s, %s, %s)"
        inserted = self.db.execute(insert_sql, players_to_insert, many=True)
        
        return inserted

# ------------------------------ Monte Carlo ------------------------------
# ------------------------------ Trading ------------------------------
# Init
fill_players_data()