import numpy as np
from sklearn.linear_model import Ridge
import mysql.connector
import json
from collections import defaultdict
import scipy.sparse as sp

# ---------------------------
# Database Connection
# ---------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="venomio",
    database="test"
)
cursor = connection.cursor()

cursor.execute("""
    SELECT game_off_def_id, off_players, minutes_played, total_xg 
    FROM game_data
""")
rapm_rows = cursor.fetchall()

cursor.close()
connection.close()

# ---------------------------
# Data Processing
# ---------------------------

players_set = set()
matches = defaultdict(list)

for row in rapm_rows:
    connection_id = row[0]
    players_list = json.loads(row[1])
    minutes = row[2]
    team_score = float(row[3])
    matches[connection_id].append((players_list, minutes, team_score))
    players_set.update(players_list)

players = sorted(players_set)
num_players = len(players)
players_to_index = {player: idx for idx, player in enumerate(players)}

rows = []
cols = []
data_vals = []
y = []
sample_weights = []
row_num = 0
for connection_id, teams in matches.items():
    a_players, minutes, home_score = teams[0]
    b_players, _, away_score = teams[1]

    for p in a_players:
        rows.append(row_num)
        cols.append(players_to_index[p])
        data_vals.append(1)

    for p in b_players:
        rows.append(row_num)
        cols.append(num_players + players_to_index[p])
        data_vals.append(-1)

    y.append(home_score / minutes)
    sample_weights.append(minutes)
    row_num += 1

    for p in b_players:
        rows.append(row_num)
        cols.append(players_to_index[p])
        data_vals.append(1)
    for p in a_players:
        rows.append(row_num)
        cols.append(num_players + players_to_index[p])
        data_vals.append(-1)

    y.append(away_score / minutes)
    sample_weights.append(minutes)
    row_num += 1

X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
y = np.array(y)
sample_weights = np.array(sample_weights)

# ---------------------------
# Model Training
# ---------------------------

ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg')
ridge.fit(X, y, sample_weight=sample_weights)

offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))

# ---------------------------
# Save data
# ---------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="venomio", 
    database="test"
)

cursor = connection.cursor()

for player in players:
    off = round(offensive_ratings[player], 3)
    deff = round(defensive_ratings[player], 3)
    total = off + deff
    update_query = """
    UPDATE players_data
    SET off_coef = %s, def_coef = %s, total_coef = %s
    WHERE player_name = %s;
    """
    cursor.execute(update_query, (off, deff, total, player))

connection.commit()
cursor.close()
connection.close()

print("Coefficients updated successfully! 🎯")