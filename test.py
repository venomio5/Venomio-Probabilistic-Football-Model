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

non_updated_fouls_df = core.DB.select("SELECT * FROM match_info WHERE total_fouls == 0;")

for _, row in non_updated_fouls_df.iterrows():
    match_id = row['match_id']
    print(match_id)

    match_bd_df = core.DB.select("SELECT fouls_committed, yellow_cards, red_cards FROM match_breakdown WHERE match_id = %s;", (match_id,))

    for _, row in match_bd_df.iterrows():
        match_id = row['match_id']
    current_team = player_id_df.iloc[0]['current_team']

    # core.DB.execute(
    #     "UPDATE shots_data SET team_id = %s WHERE shot_id = %s",
    #     (int(current_team), int(row['shot_id']))
    # )
