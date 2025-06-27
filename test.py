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


non_updated_shots_df = core.DB.select("SELECT * FROM shots_data WHERE team_id IS NULL;")

for _, row in non_updated_shots_df.iterrows():
    psxg = float(row['psxg'])
    outcome = np.random.poisson(psxg)

    core.DB.execute(
        "UPDATE shots_data SET outcome = %s WHERE shot_id = %s",
        (outcome, row['shot_id'])
    )
