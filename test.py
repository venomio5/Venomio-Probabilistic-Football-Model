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

s = Service('chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=s, options=options)
driver.get("https://fbref.com/en/matches/06a6419c/Atletico-Mineiro-Sao-Paulo-April-6-2025-Serie-A")
home_team = "Atlético Mineiro"
away_team = "São Paulo"
match_id = 273

home_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="a"]/table')))
home_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", home_table))

away_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="b"]/table')))
away_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", away_table))

home_players = extract_players(home_data, home_team)
away_players = extract_players(away_data, away_team)

try:
    events_wrap = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="events_wrap"]')))
except Exception as e:
    print(e)

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
    if event.find_elements(By.CSS_SELECTOR, '.goal'):
        goal_events.append((event_minute, team))
    if event.find_elements(By.CSS_SELECTOR, '.red_card'):
        player_links = event.find_elements(By.CSS_SELECTOR, 'a')
        if player_links:
            player_name = player_links[0].text.strip()
            red_events.append((event_minute, player_name, team))
            print(event_minute, player_name, team)

total_minutes = 90 + extra_first_half + extra_second_half

shots_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="shots_all"]')))
shots_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", shots_table))
shots_df = shots_data[0]
shots_df.columns = pd.MultiIndex.from_tuples(shots_df.columns)
selected_columns = shots_df.loc[:, [('Unnamed: 0_level_0', 'Minute'),
                                    ('Unnamed: 2_level_0', 'Squad'),
                                    ('Unnamed: 3_level_0', 'xG'),
                                    ('Unnamed: 7_level_0', 'Body Part')]]
cleaned_columns = []
for col in selected_columns.columns:
    if 'Unnamed' in col[0]:
        cleaned_columns.append(col[1])
    else:
        cleaned_columns.append('_'.join(col).strip())
selected_columns.columns = cleaned_columns
selected_columns = selected_columns[selected_columns['Minute'].notna() & (selected_columns['Minute'] != 'Minute')]
selected_columns['Minute'] = selected_columns['Minute'].astype(str).str.replace(r'\+.*', '', regex=True).str.strip().astype(float).astype(int)
selected_columns['xG'] = selected_columns['xG'].astype(float)

event_minutes = [se[0] for se in subs_events] + [ge[0] for ge in goal_events] + [re[0] for re in red_events]
standard_boundaries = [0, 15, 30, 45, 60, 75]
boundaries = sorted(set(standard_boundaries) | set(event_minutes) | {total_minutes})

for seg_start, seg_end in zip(boundaries, boundaries[1:]):
    seg_duration = seg_end - seg_start

    teamA_lineup = get_lineups(home_players, subs_events, seg_start, "home", red_events)
    teamB_lineup = get_lineups(away_players, subs_events, seg_start, "away", red_events)

    seg_shots = selected_columns[(selected_columns['Minute'] >= seg_start) & (selected_columns['Minute'] < seg_end)]
    teamA_headers = 0
    teamA_footers = 0
    teamA_hxg = 0.0
    teamA_fxg = 0.0
    teamB_headers = 0
    teamB_footers = 0
    teamB_hxg = 0.0
    teamB_fxg = 0.0
    for _, row in seg_shots.iterrows():
        if home_team in row['Squad']:
            if "Head" in row['Body Part']:
                teamA_headers += 1
                teamA_hxg += row['xG']
            elif "Foot" in row['Body Part']:
                teamA_footers += 1
                teamA_fxg += row['xG']
        elif away_team in row['Squad']:
            if "Head" in row['Body Part']:
                teamB_headers += 1
                teamB_hxg += row['xG']
            elif "Foot" in row['Body Part']:
                teamB_footers += 1
                teamB_fxg += row['xG']

    cum_goal_home = sum(1 for minute, t in goal_events if minute <= seg_end and t == "home")
    cum_goal_away = sum(1 for minute, t in goal_events if minute <= seg_end and t == "away")

    goal_diff = cum_goal_home - cum_goal_away
    if goal_diff == 0:
        match_state = "0"
    elif goal_diff == 1:
        match_state = "1"
    elif goal_diff > 1:
        match_state = "1.5"
    elif goal_diff == -1:
        match_state = "-1"
    else:
        match_state = "-1.5"

    cum_red_home = sum(1 for minute, _, t in red_events if minute <= seg_end and t == "home")
    cum_red_away = sum(1 for minute, _, t in red_events if minute <= seg_end and t == "away")
    print(f"home red cards = {cum_red_home}")
    print(f"away red cards = {cum_red_away}")
    red_diff = cum_red_away - cum_red_home
    print(f"{red_diff} = {cum_red_home} - {cum_red_away}")
    if red_diff == 0:
        player_dif = "0"
    elif red_diff == 1:
        player_dif = "1"
    elif red_diff > 1:
        player_dif = "1.5"
    elif red_diff == -1:
        player_dif = "-1"
    else:
        player_dif = "-1.5"

    match_segment = min((seg_start // 15) + 1, 6)

    sql = "INSERT IGNORE INTO match_detail (match_id, teamA_players, teamB_players, teamA_headers, teamA_footers, teamA_hxg, teamA_fxg, teamB_headers, teamB_footers, teamB_hxg, teamB_fxg, minutes_played, match_state, match_segment, player_dif) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    params = (match_id, json.dumps(teamA_lineup, ensure_ascii=False), json.dumps(teamB_lineup, ensure_ascii=False), teamA_headers, teamA_footers, teamA_hxg, teamA_fxg, teamB_headers, teamB_footers, teamB_hxg, teamB_fxg, seg_duration, match_state, match_segment, player_dif)
    print(sql, params)