import xgboost as xgb
import mysql.connector
import pandas as pd
import numpy as np

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
    SELECT off_rapm, off_team, def_team, off_home, away_travel, away_elev_dif, off_status, momentum, time_segment, off_rest, def_rest, game_time, game_season, total_xg
    FROM game_data;
""")
data = cursor.fetchall()

cursor.close()
connection.close()

# ---------------------------
# Data Preprocessing
# ---------------------------
columns = ['off_rapm', 'off_team', 'def_team', 'off_home', 'away_travel', 'away_elev_dif', 'off_status', 'momentum', 'time_segment', 'off_rest', 'def_rest', 'game_time', 'game_season', 'total_xg']
df = pd.DataFrame(data, columns=columns)
df['off_rapm'] = df['off_rapm'].astype(float)
df['total_xg'] = df['total_xg'].astype(float)
df['off_home'] = df['off_home'].astype(int)
df['momentum'] = df['momentum'].astype(int)

categorical_cols = ['off_team', 'def_team', 'away_travel', 'away_elev_dif', 'off_status', 'time_segment', 'off_rest', 'def_rest', 'game_time', 'game_season']
for col in categorical_cols:
    df[col] = df[col].str.lower()

df_dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
dummy_columns = df_dummies.columns

X = pd.concat([df[['off_home', 'momentum']].reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
X[['off_home', 'momentum']] = X[['off_home', 'momentum']].astype(bool)
y = df['total_xg']

dtrain = xgb.DMatrix(X, label=y, base_margin=np.log(df['off_rapm']))

# ---------------------------
# Poisson Regression with XGBoost
# ---------------------------
params = {
    'objective': 'count:poisson',  # Poisson regression for count data
    'max_depth': 6,                # Maximum depth of a tree
    'eta': 0.05 ,                    # Learning rate
    'subsample': 0.8,              # Subsample ratio of the training instances
    'colsample_bytree': 0.8,       # Subsample ratio of columns when constructing each tree
    'min_child_weight': 5,       # restricts number of samples per leaf, avoid overfit
    'gamma': 0,                  # require a minimum gain to make a further partition (helps trees generalize) 2
    'lambda': 1,                 # L2 regularization
    'alpha': 0,                  # use L1 if you have lots of sparse/dummy vars
    'tree_method': 'hist',       # speeds up training, EXCELLENT for CPUs
}

num_round = 500  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_round)

# ---------------------------
# Predicting the Next Match
# ---------------------------
new_match = pd.DataFrame({
    'off_rapm': [0.8300],
    'off_team': ['Bayern'],
    'def_team': ['Milan'],
    'off_home': [True],
    'away_travel': ['Cross Country'],
    'away_elev_dif': ['Higher'],
    'off_status': ['Level'],
    'momentum': [False],
    'time_segment': ['Q1'],
    'off_rest': ['Normal'],
    'def_rest': ['Normal'],
    'game_time': ['Night'],
    'game_season': ['Fall'],
})
baseline_new = np.log(new_match.pop('off_rapm'))
for col in categorical_cols:
    new_match[col] = new_match[col].str.lower()
new_match_dummies = pd.get_dummies(new_match[categorical_cols], prefix=categorical_cols)
new_match_dummies = new_match_dummies.reindex(columns=dummy_columns, fill_value=0)
new_X = pd.concat([new_match[['off_home', 'momentum']].reset_index(drop=True),
                   new_match_dummies.reset_index(drop=True)], axis=1)
new_X[['off_home', 'momentum']] = new_X[['off_home', 'momentum']].astype(bool)

# Predict the points for the next match
dnext_match = xgb.DMatrix(new_X, base_margin=baseline_new)
predicted_points = bst.predict(dnext_match)
print(f"Predicted points for the next match: {predicted_points[0]}")

import matplotlib.pyplot as plt
xgb.plot_tree(bst, num_trees=2)
xgb.plot_importance(bst)
plt.show()