from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable, Sequence
import pandas as pd
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import requests
import math
from fuzzywuzzy import process, fuzz
import re
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import json
from tqdm import tqdm
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QToolButton, QMenu
import core
import ast
import xgboost as xgb
import matplotlib.pyplot as plt

def train_context_ras_model():
    def flip(series: pd.Series) -> pd.Series:
        flipped = -series
        flipped[series == 0] = 0.0             # remove the -0.0 sign bit
        return flipped
    
    sql_query = f"""
        SELECT 
            mi.match_id,
            mi.home_elevation_dif,
            mi.away_elevation_dif,
            mi.away_travel,
            mi.home_rest_days,
            mi.away_rest_days,
            mi.temperature_c,
            mi.is_raining,
            mi.date,
            md.teamA_pdras,
            md.teamB_pdras,
            md.match_state,
            md.match_segment,
            md.minutes_played,
            md.player_dif,
            (md.teamA_headers + md.teamA_footers) AS home_shots,
            (md.teamB_headers + md.teamB_footers) AS away_shots
        FROM match_info mi
        JOIN match_detail md ON mi.match_id = md.match_id
    """
    context_df = core.DB.select(sql_query)
    context_df['date'] = pd.to_datetime(context_df['date'])
    context_df['match_state'] = pd.to_numeric(context_df['match_state'], errors='raise').astype(float)
    context_df['player_dif']  = pd.to_numeric(context_df['player_dif'],  errors='raise').astype(float)

    def _bucket(ts):
        h = ts.hour
        if 9 <= h < 14:
            return 'aft'
        if 14 <= h < 19:
            return 'evening'
        return 'night'

    home_df = pd.DataFrame({
        'shots'              : context_df['home_shots'],
        'total_ras'          : context_df['teamA_pdras'],
        'minutes_played'     : context_df['minutes_played'],
        'team_is_home'       : 1,
        'team_elevation_dif' : context_df['home_elevation_dif'],
        'opp_elevation_dif'  : context_df['away_elevation_dif'],
        'team_travel'        : 0,
        'opp_travel'         : context_df['away_travel'],
        'team_rest_days'     : context_df['home_rest_days'],
        'opp_rest_days'      : context_df['away_rest_days'],
        'match_state'        : context_df['match_state'],
        'match_segment'      : context_df['match_segment'],
        'player_dif'         : context_df['player_dif'],
        'temperature_c'      : context_df['temperature_c'],
        'is_raining'         : context_df['is_raining'],
        'match_time'         : context_df['date'].apply(_bucket)
    })

    away_df = pd.DataFrame({
        'shots'              : context_df['away_shots'],
        'total_ras'          : context_df['teamB_pdras'],
        'minutes_played'     : context_df['minutes_played'],
        'team_is_home'       : 0,
        'team_elevation_dif' : context_df['away_elevation_dif'],
        'opp_elevation_dif'  : context_df['home_elevation_dif'],
        'team_travel'        : context_df['away_travel'],
        'opp_travel'         : 0,
        'team_rest_days'     : context_df['away_rest_days'],
        'opp_rest_days'      : context_df['home_rest_days'],
        'match_state'        : flip(context_df['match_state']),
        'match_segment'      : context_df['match_segment'],
        'player_dif'         : flip(context_df['player_dif']),
        'temperature_c'      : context_df['temperature_c'],
        'is_raining'         : context_df['is_raining'],
        'match_time'         : context_df['date'].apply(_bucket)
    })

    df = pd.concat([home_df, away_df], ignore_index=True)

    df['shots_per_min']     = df['shots']      / df['minutes_played']
    df['ras_per_min']       = df['total_ras']  / df['minutes_played']

    cat_cols  = ['match_state', 'match_segment', 'player_dif', 'match_time']
    bool_cols = ['team_is_home', 'is_raining']
    num_cols  = ['team_elevation_dif', 'opp_elevation_dif', 'team_travel', 'opp_travel', 'team_rest_days', 'opp_rest_days', 'temperature_c']
    
    required_cols = cat_cols + bool_cols + num_cols + ['shots', 'total_ras']
    missing_cols  = [c for c in ['shots', 'total_ras'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing expected columns: {missing_cols}')

    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    for c in cat_cols:
        df[c] = df[c].astype(str).str.lower()

    df[bool_cols] = df[bool_cols].astype(int)

    X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols)
    X     = pd.concat([df[num_cols], df[bool_cols], X_cat], axis=1)

    y           = df['shots_per_min']
    base_margin = np.log(df['ras_per_min'].clip(lower=1e-6))

    dtrain = xgb.DMatrix(X, label=y, base_margin=base_margin)

    params = dict(objective='count:poisson',
                    tree_method='hist',
                    max_depth=6,
                    eta=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5)

    booster = xgb.train(params, dtrain, num_boost_round=300)
    return booster, X.columns

def predict_next_match(booster, feature_columns, new_match):
    categorical_cols = ['match_state', 'match_segment', 'player_dif', 'match_time']
    bool_cols = ['team_is_home', 'is_raining']
    num_cols = ['team_elevation_dif', 'opp_elevation_dif', 'team_travel', 'opp_travel', 'team_rest_days', 'opp_rest_days', 'temperature_c']
    
    base_margin = np.log(new_match.pop('total_ras').clip(lower=1e-6))

    for col in categorical_cols:
        new_match[col] = new_match[col].astype(str).str.lower()
    new_match[bool_cols] = new_match[bool_cols].astype(int)

    new_cat = pd.get_dummies(new_match[categorical_cols], prefix=categorical_cols)
    new_cat = new_cat.reindex(columns=[col for col in feature_columns if any(col.startswith(prefix + "_") for prefix in categorical_cols)], fill_value=0)

    new_X = pd.concat([
        new_match[num_cols + bool_cols].reset_index(drop=True),
        new_cat.reset_index(drop=True)
    ], axis=1)

    new_X = new_X.reindex(columns=feature_columns, fill_value=0)

    dmatrix = xgb.DMatrix(new_X, base_margin=base_margin)
    prediction = booster.predict(dmatrix)
    return prediction[0]

booster, columns = train_context_ras_model()
# xgb.plot_importance(booster, importance_type='gain')
# plt.title("Feature Importance (Gain)")
# plt.tight_layout()
# plt.show()

next_match = pd.DataFrame([{
    'shots': 0,  # Placeholder, not used in prediction
    'total_ras': 0.07,
    'team_is_home': 1,
    'team_elevation_dif': 208,
    'opp_elevation_dif': 569,
    'team_travel': 0,
    'opp_travel': 2370,
    'team_rest_days': 6,
    'opp_rest_days': 6,
    'match_state': 0.0,
    'match_segment': 1,
    'player_dif': 0.0,
    'temperature_c': 16.0,
    'is_raining': 0,
    'match_time': 'evening'
}])

pred = predict_next_match(booster, columns, next_match)
print(f"Predicted shots: {pred:.3f}")