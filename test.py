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
from xgboost import plot_importance
from xgboost import plot_tree


def train_refined_sq_model() -> tuple[xgb.Booster, list[str]]:  
    sql = """
        SELECT
            total_plsqa,
            shooter_sq,
            assister_sq,
            CASE
                WHEN match_state <  0 THEN 'Trailing'
                WHEN match_state =  0 THEN 'Level'
                ELSE                       'Leading'
            END               AS match_state,
            CASE
                WHEN player_dif <  0 THEN 'Neg'
                WHEN player_dif =  0 THEN 'Neu'
                ELSE                       'Pos'
            END               AS player_dif,
            xg
        FROM shots_data
        WHERE total_plsqa IS NOT NULL
    """
    
    df = core.DB.select(sql)
    
    cat_cols = ['match_state', 'player_dif']
    num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    
    for c in cat_cols:
        df[c] = df[c].astype(str)
    
    X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)
    X     = pd.concat([df[num_cols], X_cat], axis=1).astype(float)
    y     = df["xg"].astype(float)
    
    dtrain = xgb.DMatrix(X, label=y)
    
    params = dict(
        objective        = "reg:logistic",
        eval_metric      = "rmse",
        tree_method      = "hist",
        max_depth        = 6,
        eta              = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 2
    )
    
    booster = xgb.train(params, dtrain, num_boost_round=400)
    return booster, X.columns.tolist()

def predict_refined_sq(booster        : xgb.Booster,
                       feature_columns: list[str],
                       shot_features  : dict,
                       *,
                       raw            : bool = False) -> float:
    
    cat_cols = ['match_state', 'player_dif']
    num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']
    
    row = shot_features.copy()
    
    for c in cat_cols:
        row[c] = str(row[c]).title()
    
    num_df = pd.DataFrame([{k: row[k] for k in num_cols}])
    cat_df = pd.get_dummies(pd.DataFrame([{c: row[c] for c in cat_cols}]), prefix=cat_cols)
    cat_df = cat_df.reindex(
        columns=[c for c in feature_columns if any(c.startswith(p + '_') for p in cat_cols)],
        fill_value=0
    )
    
    X = (
        pd.concat([num_df, cat_df], axis=1)
          .reindex(columns=feature_columns, fill_value=0)
          .astype(float)
    )
    
    dmat = xgb.DMatrix(X)
    pred = booster.predict(dmat, output_margin=raw)
    return float(pred[0])

booster, rsq_features = train_refined_sq_model()

non_updated_matches_df = core.DB.select("SELECT * FROM shots_data WHERE total_PLSQA IS NULL OR RSQ IS NULL;")

non_updated_matches_df['off_players'] = non_updated_matches_df['off_players'].apply(
    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
)
non_updated_matches_df['def_players'] = non_updated_matches_df['def_players'].apply(
    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
)

players_needed = set()
for _, row in non_updated_matches_df.iterrows():
    players_needed.update(row['off_players'])
    players_needed.update(row['def_players'])

if players_needed:
    placeholders = ','.join(['%s'] * len(players_needed))
    players_sql = (
        f"SELECT player_id, off_hxg_coef, def_hxg_coef, off_fxg_coef, def_fxg_coef, headers, footers, key_passes, hxg, fxg, kp_hxg, kp_fxg, hpsxg, fpsxg, gk_psxg, gk_ga "
        f"FROM players_data "
        f"WHERE player_id IN ({placeholders});"
    )
    players_data_df  = core.DB.select(players_sql, list(players_needed))
    p_dict = players_data_df.set_index("player_id").to_dict("index")
else:
    p_dict = {}

for _, row in non_updated_matches_df.iterrows():
    off_ids = row['off_players']
    def_ids = row['def_players']
    bp = row['shot_type']
    shooter_id = row['shooter_id']
    assister_id = row['assister_id']

    if bp == "head":
        offense = sum(p_dict.get(pid, {}).get('off_hxg_coef', 0) for pid in off_ids)
        defense = sum(p_dict.get(pid, {}).get('def_hxg_coef', 0) for pid in def_ids)
    else:
        offense = sum(p_dict.get(pid, {}).get('off_fxg_coef', 0) for pid in off_ids)
        defense = sum(p_dict.get(pid, {}).get('def_fxg_coef', 0) for pid in def_ids)

    plsqa = offense - defense

    shooter_data = p_dict.get(shooter_id, {})
    if bp == "head":
        numerator = shooter_data.get('hxg', 0)
        denominator = shooter_data.get('headers', 1)
        shooter_A = shooter_data.get('hpsxg', 0) / numerator if numerator else 0.0
    else:
        numerator = shooter_data.get('fxg', 0)
        denominator = shooter_data.get('footers', 1)
        shooter_A = shooter_data.get('fpsxg', 0) / numerator if numerator else 0.0

    shooter_sq = numerator / denominator if denominator else 0.0

    if not assister_id or not isinstance(p_dict[assister_id], dict):
        assister_sq = None
    else:
        assister_data = p_dict.get(assister_id, {})
        if bp == "head":
            numerator = assister_data.get('kp_hxg', 0)
        else:
            numerator = assister_data.get('kp_fxg', 0)
        denominator = assister_data.get('key_passes', 1)
        assister_sq = numerator / denominator if denominator else 0.0

    gk_A = None
    for pid in def_ids:
        gk_data = p_dict.get(pid, {})
        gk_psxg = gk_data.get('gk_psxg')
        gk_ga = gk_data.get('gk_ga')
        if gk_psxg is not None and gk_ga is not None and gk_psxg != 0:
            gk_A = 1.0 - (gk_ga / gk_psxg)
            break

    rsq = predict_refined_sq(
        booster,
        rsq_features,
        dict(
            total_plsqa = plsqa,
            shooter_sq  = shooter_sq,
            assister_sq = assister_sq,
            match_state = row['match_state'],
            player_dif  = row['player_dif']
        )
    )

    core.DB.execute(
        "UPDATE shots_data SET total_PLSQA = %s, shooter_SQ = %s, assister_SQ = %s, RSQ = %s, shooter_A = %s, GK_A = %s WHERE shot_id = %s",
        (plsqa, shooter_sq, assister_sq, rsq, shooter_A, gk_A, row['shot_id'])
    )