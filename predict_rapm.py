import mysql.connector
import json
import sys

# ---------------------------
# Load players and each coeficient
# ---------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="venomio",
    database="test"
)
cursor = connection.cursor()

cursor.execute("""
    SELECT off_players, def_players, minutes_played
    FROM game_data
""")
players_rows = cursor.fetchall()

cursor.execute("SELECT player_name, off_coef, def_coef FROM players_data")
rapm_rows = cursor.fetchall()

cursor.close()
connection.close()

off_dict = {name: off for name, off, _ in rapm_rows}
def_dict = {name: deff for name, _, deff in rapm_rows}

for row in players_rows:
    off_players = json.loads(row[0])
    def_players = json.loads(row[1])
    minutes = int(row[2])
    predicted_xg = (sum(off_dict.get(p, 0) for p in off_players) - sum(def_dict.get(p, 0) for p in def_players)) * minutes

    print(f"Predicted xG: {round(predicted_xg, 3)}")