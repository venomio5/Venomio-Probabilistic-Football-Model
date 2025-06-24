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

def extract_players(data, team_name):
    team_initials = ''.join(word[0].upper() for word in team_name.split() if word)
    df = data[0]
    filtered = df[~df.iloc[:, 1].str.contains("Bench", na=False)]
    players = [
        f"{row[1]}_{row[0]}_{team_initials}"
        for _, row in filtered.iloc[:, :2].iterrows()
    ]
    return players

def initialize_player_stats(player_list):
    return {
        player: {
            "player_id": player,
            "starter": i < 11,
            "headers": 0,
            "footers": 0,
            "key_passes": 0,
            "non_assisted_footers": 0,
            "hxg": 0.0,
            "fxg": 0.0,
            "kp_hxg": 0.0,
            "kp_fxg": 0.0,
            "hpsxg": 0.0,
            "fpsxg": 0.0,
            "gk_psxg": 0.0,
            "gk_ga": 0
        } for i, player in enumerate(player_list)
    }

def get_lineups(initial_players, sub_events, current_minute, team, red_events=None):
    if red_events is None:
        red_events = []
    
    roster_mapping = {}
    for player in initial_players:
        key = player.split("_")[0]
        roster_mapping[key] = player

    lineup = initial_players[:11]

    filtered_subs = [s for s in sub_events if s[3] == team]
    filtered_subs = sorted(filtered_subs, key=lambda x: x[0])

    for sub_minute, player_out, player_in, _ in filtered_subs:
        if sub_minute > current_minute:
            break

        for idx, player in enumerate(lineup):
            if player.split("_")[0] == player_out:
                replacement = roster_mapping.get(player_in, player_in)
                lineup[idx] = replacement
                roster_mapping[player_in] = replacement
                break

    sent_off = [p for m, p, t in red_events if t == team and m <= current_minute]
    lineup = [p for p in lineup if p.split("_")[0] not in sent_off]

    lineup = [roster_mapping.get(p.split("_")[0], p) for p in lineup]
    return lineup

# Match detail
s = Service('chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=s, options=options)
driver.get('https://fbref.com/en/matches/25ce23b0/Sao-Paulo-Fortaleza-May-2-2025-Serie-A')
home_team = "São Paulo"
away_team = "Fortaleza"

home_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="a"]/table')))
home_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", home_table))

away_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="b"]/table')))
away_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", away_table))

home_players = extract_players(home_data, home_team)
away_players = extract_players(away_data, away_team)

events_wrap = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="events_wrap"]')))

subs_events = []
goal_events = []
red_events = []
extra_first_half = 0
extra_second_half = 0

for event in events_wrap.find_elements(By.CSS_SELECTOR, '.event'):
    lines = event.text.strip().split('\n')
    event_minute = None
    for line in lines:
        m = re.match(r'^\s*(\d+)(?:\+(\d+))?’.?$', line)
        if m:
            base_minute = int(m.group(1))
            plus = m.group(2)
            if base_minute == 45 and plus:
                extra_first_half = max(extra_first_half, int(plus))
            if base_minute == 90 and plus:
                extra_second_half = max(extra_second_half, int(plus))
            event_minute = base_minute if not plus else base_minute + int(plus)
            break
    if event_minute is None:
        continue
    classes = event.get_attribute("class").split()
    team = "home" if "a" in classes else "away" if "b" in classes else None
    if team is None:
        continue
    if event.find_elements(By.CSS_SELECTOR, '.substitute_in'):
        player_out, player_in = None, None
        for line in lines:
            if not re.match(r'^\s*\d+(?:\+\d+)?’$', line) and not re.match(r'^\s*\d+\s*:\s*\d+\s*$', line):
                if not line.startswith("for "):
                    player_in = line.strip()
                else:
                    player_out = line[len("for "):].strip()
        if player_out is not None and player_in is not None:
            subs_events.append((event_minute, player_out, player_in, team))
    if event.find_elements(By.CSS_SELECTOR, '.goal, .own_goal'):
        goal_events.append((event_minute, team))
    if event.find_elements(By.CSS_SELECTOR, '.red_card'):
        player_links = event.find_elements(By.CSS_SELECTOR, 'a')
        if player_links:
            player_name = player_links[0].text.strip()
            red_events.append((event_minute, player_name, team))

total_minutes = 90 + extra_first_half + extra_second_half

# Match breakdown
print(f'\n{subs_events}')
home_player_stats = initialize_player_stats(home_players)
away_player_stats = initialize_player_stats(away_players)

for sub_event in subs_events:
    sub_minute, player_out, player_in, sub_team = sub_event
    if sub_team == "home":
        home_goals = sum(1 for minute, t in goal_events if t == "home" and minute <= sub_minute)
        away_goals = sum(1 for minute, t in goal_events if t == "away" and minute <= sub_minute)
        state = "leading" if home_goals > away_goals else "level" if home_goals == away_goals else "trailing"
        for key in home_player_stats:
            if key.split("_")[0] == player_out:
                home_player_stats[key]["sub_out_min"] = sub_minute
                home_player_stats[key]["out_status"] = state
        found_in = False
        for key in home_player_stats:
            if key.split("_")[0] == player_in:
                home_player_stats[key]["sub_in_min"] = sub_minute
                home_player_stats[key]["in_status"] = state
                found_in = True
        if not found_in:
            home_player_stats[player_in] = {"starter": False, "sub_in_min": sub_minute, "in_status": state}
    else:
        away_goals = sum(1 for minute, t in goal_events if t == "away" and minute <= sub_minute)
        home_goals = sum(1 for minute, t in goal_events if t == "home" and minute <= sub_minute)
        state = "leading" if away_goals > home_goals else "level" if away_goals == home_goals else "trailing"
        for key in away_player_stats:
            if key.split("_")[0] == player_out:
                away_player_stats[key]["sub_out_min"] = sub_minute
                away_player_stats[key]["out_status"] = state
        found_in = False
        for key in away_player_stats:
            if key.split("_")[0] == player_in:
                away_player_stats[key]["sub_in_min"] = sub_minute
                away_player_stats[key]["in_status"] = state
                found_in = True
        if not found_in:
            away_player_stats[player_in] = {"starter": False, "sub_in_min": sub_minute, "in_status": state}

final_home_goals = sum(1 for minute, t in goal_events if t == "home" and minute <= total_minutes)
final_away_goals = sum(1 for minute, t in goal_events if t == "away" and minute <= total_minutes)

for key, stat in list(home_player_stats.items()):

    stat.setdefault("starter", True)

    stat.setdefault("sub_in_min",  None)
    stat.setdefault("sub_out_min", None)
    stat.setdefault("in_status",   None)
    stat.setdefault("out_status",  None)

    in_min  = 0 if stat["starter"] else stat["sub_in_min"]
    out_min = stat["sub_out_min"] if stat["sub_out_min"] is not None else total_minutes
    out_min = min(out_min, 90)

    stat["minutes_played"] = out_min - (in_min if in_min is not None else 0)

for key, stat in list(away_player_stats.items()):

    stat.setdefault("starter", True)

    stat.setdefault("sub_in_min",  None)
    stat.setdefault("sub_out_min", None)
    stat.setdefault("in_status",   None)
    stat.setdefault("out_status",  None)

    in_min  = 0 if stat["starter"] else stat["sub_in_min"]
    out_min = stat["sub_out_min"] if stat["sub_out_min"] is not None else total_minutes
    out_min = min(out_min, 90)

    stat["minutes_played"] = out_min - (in_min if in_min is not None else 0)

for key in list(home_player_stats.keys()):
    print(home_player_stats[key])