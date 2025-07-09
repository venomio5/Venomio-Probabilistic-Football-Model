from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable, Sequence
import pandas as pd
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime, date, time, timedelta
import numpy as np
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import requests
import math
from rapidfuzz import process, fuzz
import re
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import json
from tqdm import tqdm
import ast
import xgboost as xgb
import multiprocessing
import os
import itertools 
import copy
import unicodedata

# --------------- Useful Classes, Functions & Variables ---------------
class DatabaseManager:
    """
    Optimizes the initializaiton of a MySQLConnectionPool with UTF-8MB4 encoding.

    Usage Example:
    db = DatabaseManager(
        host="localhost",
        user="admin",
        password="secret",
        database="production"
    )

    Select example
    df = db.select("SELECT * FROM users WHERE status = %s", ("active",))

    Insert example
    affected = db.execute("INSERT INTO logs (event) VALUES (%s)", ("startup",))

    Batch insert
    affected = db.execute(
        "INSERT INTO metrics (key, value) VALUES (%s, %s)",
        [("cpu", 0.93), ("ram", 0.72)],
        many=True
    )
    """
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        pool_name: str = "db_pool",
        pool_size: int = 6,
    ) -> None:
        self._pool: MySQLConnectionPool = MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            autocommit=False,
        )

    @contextmanager
    def _connection(self):
        conn = self._pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _cursor(self, conn):
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def select(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        with self._connection() as conn, self._cursor(conn) as cur:
            cur.execute(sql, params or ())
            columns = [c[0] for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        many: bool = False,
    ) -> int:
        with self._connection() as conn, self._cursor(conn) as cur:
            if many and isinstance(params, Iterable):
                cur.executemany(sql, params)  # type: ignore[arg-type]
            else:
                cur.execute(sql, params or ())
            return cur.rowcount

class Fill_Teams_Data:
    """
    - Fetches the fixture URL from the league_data table.
    - Scrapes team names, venues, and fixture URLs using Selenium.
    - Resolves each venue to precise geographic coordinates via Nominatim.
    - Retrieves elevation data for each location via Open-Elevation API.
    - Extracts direct 'Scores & Fixtures' URLs for each team.
    - Inserts or updates team_data table with enriched team metadata.
    - Removes obsolete team entries for the league. 
    """
    def __init__(self, league_id):
        self.league_id = int(league_id)
        league_df = DB.select(f"SELECT league_id, fbref_fixtures_url FROM league_data WHERE league_id = {self.league_id}")
        league_url = league_df['fbref_fixtures_url'].values[0]

        teams_dict = self.get_teams(league_url)
        insert_data = []

        for team, (venue, team_page_url) in teams_dict.items():
            lat, lon = self.get_coordinates(team, venue)
            coordinates_str = f"{lat},{lon}"
            elevation = self.get_elevation(lat, lon)
            fixtures_url = self.get_scores_fixtures_url(team_page_url)
            insert_data.append((team, elevation, coordinates_str, fixtures_url, self.league_id))

        DB.execute(
            """
            INSERT INTO team_data (team_name, team_elevation, team_coordinates, team_fixtures_url, league_id)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                team_elevation = VALUES(team_elevation),
                team_coordinates = VALUES(team_coordinates),
                team_fixtures_url  = VALUES(team_fixtures_url)
            """,
            insert_data,
            many=True
        )

        current_team_names = tuple(teams_dict.keys())

        if current_team_names:
            placeholders = ','.join(['%s'] * len(current_team_names))
            DB.execute(
                f"""
                DELETE FROM team_data
                WHERE league_id = %s AND team_name NOT IN ({placeholders})
                """,
                (self.league_id, *current_team_names)
            )

    def get_teams(self, url):
        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(url)
        driver.execute_script("window.scrollTo(0, 1000);")

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.stats_table")))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
        team_venue_map = {}

        for row in rows:
            try:
                home_element = row.find_element(By.CSS_SELECTOR, "[data-stat='home_team']")
                venue_element = row.find_element(By.CSS_SELECTOR, "[data-stat='venue']")
                anchor        = home_element.find_element(By.TAG_NAME, "a")  
                home_team     = anchor.text.strip()
                team_page_url = anchor.get_attribute("href")
                venue         = venue_element.text.strip()

                if home_team == "Home":
                    continue

                if home_team and home_team not in team_venue_map:
                    team_venue_map[home_team] = (venue, team_page_url)

            except NoSuchElementException:
                continue
        driver.quit()
        
        return team_venue_map

    def get_coordinates(self, team, place_name):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'GeoDataScript/1.0'
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            refined_input = input(f"No coordinates found for '{team}: {place_name}'\nEnter a more precise address for this location: ").strip()
            return self.get_coordinates(team, refined_input)
        
        print(f"{team}: {data[0]['display_name']}")

        latitude = float(data[0]['lat'])
        longitude = float(data[0]['lon'])
        return latitude, longitude

    def get_elevation(self, latitude, longitude):
        url = "https://api.open-elevation.com/api/v1/lookup"
        params = {
            "locations": f"{latitude},{longitude}"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or not data['results']:
            raise ValueError("No elevation data returned.")

        elevation_meters = data['results'][0]['elevation']
        return int(elevation_meters)

    def get_scores_fixtures_url(self, team_page_url):
        s = Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(team_page_url)

        nav  = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "inner_nav"))
        )
        link = nav.find_element(By.XPATH, ".//a[normalize-space(text())='Scores & Fixtures']")
        fixtures_url = link.get_attribute("href")

        driver.quit()
        return fixtures_url

DB = DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

def get_team_name_by_id(team_id):
    query = "SELECT team_name FROM team_data WHERE team_id = %s"
    result = DB.select(query, (team_id,))
    if not result.empty:
        return result.iloc[0]["team_name"]
    return None

def get_team_id_by_name(team_name):
    query = "SELECT team_id FROM team_data WHERE team_name = %s"
    result = DB.select(query, (team_name,))
    if not result.empty:
        return int(result.iloc[0]["team_id"])
    return None

def get_league_name_by_id(league_id):
    query = "SELECT league_name FROM league_data WHERE league_id = %s"
    result = DB.select(query, (league_id,))
    if not result.empty:
        return result.iloc[0]["league_name"]
    return None

def match_players(team_id, raw_source):
    if hasattr(raw_source, "toPlainText"):
        raw_text = raw_source.toPlainText()
        raw_list = [line.strip() for line in raw_text.split("\n")]
    elif isinstance(raw_source, (list, tuple)):
        raw_list = [str(line).strip() for line in raw_source]
    else:
        raise TypeError("raw_source must be QTextEdit-like or list/tuple")

    clean_list = [line for line in raw_list if line and not any(char.isdigit() for char in line)]

    unmatched_starters = clean_list[:11]
    unmatched_benchers = clean_list[11:]

    player_sql_query = """
        SELECT DISTINCT player_id
        FROM players_data
        WHERE current_team = %s;
    """
    players_df = DB.select(player_sql_query, (team_id, ))

    db_players = players_df['player_id'].tolist()

    matched_starters = []
    matched_benchers = []

    threshold = 80
    while unmatched_starters:
        remaining_players = []
        for player in unmatched_starters:
            closest_match = process.extractOne(player, db_players, score_cutoff=threshold)
            if closest_match:
                matched_starters.append(closest_match[0])
                db_players.remove(closest_match[0])
            else:
                remaining_players.append(player)
        if not remaining_players:
            break
        unmatched_starters = remaining_players
        threshold -= 20
    threshold = 80
    while unmatched_benchers:
        remaining_players = []
        for player in unmatched_benchers:
            closest_match = process.extractOne(player, db_players, score_cutoff=threshold)
            if closest_match:
                matched_benchers.append(closest_match[0])
                db_players.remove(closest_match[0])
            else:
                remaining_players.append(player)
        if not remaining_players:
            break
        unmatched_benchers = remaining_players
        threshold -= 20

    return matched_starters, matched_benchers

def get_referee_name(schedule_id):
    sql_query = "SELECT referee_name FROM schedule_data WHERE schedule_id = %s"
    result = DB.select(sql_query, (schedule_id,))
    return result.iloc[0]["referee_name"] if not result.empty else ""

def send_referee_name_to_db(referee_raw_name, schedule_id):
    sql_get_names = "SELECT DISTINCT referee_name FROM referee_data"
    referee_rows = DB.select(sql_get_names)
    all_names = referee_rows["referee_name"].dropna().tolist()

    best_match, _ = process.extractOne(referee_raw_name, all_names, scorer=fuzz.WRatio)

    sql = "UPDATE schedule_data SET referee_name = %s WHERE schedule_id = %s"
    DB.execute(sql, (best_match, schedule_id))

def send_lineup_to_db(players_list, schedule_id, team):
    column_name = f"{team}_players_data"
    sql_query = f"UPDATE schedule_data SET {column_name} = %s WHERE schedule_id = %s"
    DB.execute(sql_query, (json.dumps(players_list, ensure_ascii=False), schedule_id))

def get_saved_lineup(schedule_id, team):
    column_name = f"{team}_players_data"
    sql_query = f"SELECT {column_name} FROM schedule_data WHERE schedule_id = %s"
    result = DB.select(sql_query, (schedule_id,))

    if result.empty:
        return []

    raw_value = result.iloc[0][column_name]

    if not raw_value:
        return []

    try:
        return json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []

def _bucket_time(ts):

    if ts is None or pd.isna(ts):
        return 'evening'

    if isinstance(ts, time):
        ts = datetime.combine(date.today(), ts)

    if not isinstance(ts, (pd.Timestamp, datetime)):
        ts = pd.to_datetime(ts)

    h = ts.hour
    if 9 <= h < 14:
        return 'aft'
    if 14 <= h < 19:
        return 'evening'
    return 'night'

def _save(name: str, booster: xgb.Booster, columns: list[str]) -> None:
    booster.save_model(f'models/{name}_booster.json')
    with open(f'models/{name}_columns.json', 'w') as fh:
        json.dump(columns, fh)

# ------------------------------ Fetch & Remove Data ------------------------------
class UpdateSchedule:
    def __init__(self, from_date):
        self.from_date = from_date
        active_leagues_df = DB.select("SELECT * FROM league_data WHERE is_active = 1")
        
        for league_id in tqdm(active_leagues_df["league_id"].tolist(), desc="Processing leagues"):
            fbref_url = active_leagues_df[active_leagues_df['league_id'] == league_id]['fbref_fixtures_url'].values[0]
            ss_url = active_leagues_df[active_leagues_df['league_id'] == league_id]['ss_url'].values[0]
            upto_date = self.from_date + timedelta(days=5)

            games_dates, games_local_time, games_venue_time, home_teams, away_teams = self.get_games_basic_info(fbref_url, upto_date)
            smatches_dict = self.get_ss_urls(ss_url)

            for i in tqdm(range(len(games_dates)), desc="Games"):
                game_date = games_dates[i]
                game_local_time = games_local_time[i]
                game_venue_time = games_venue_time[i]

                if game_local_time is None and game_venue_time is None:
                    continue
                if game_local_time is None:
                    game_local_time = game_venue_time
                if game_venue_time is None:
                    game_venue_time = game_local_time

                home_team = home_teams[i]
                home_id = get_team_id_by_name(home_team)
                away_team = away_teams[i]
                away_id = get_team_id_by_name(away_team)

                ss_url = self.get_matched_teams_url(smatches_dict, f"{home_team} vs {away_team}")

                home_elevation_dif = self.get_team_elevation_dif(home_id, away_id, "home")
                away_elevation_dif = self.get_team_elevation_dif(home_id, away_id, "away")

                away_travel_dist = self.get_travel_distance(home_id, away_id)

                if self._schedule_exists(home_id, away_id, game_date, league_id):
                    row = DB.select(
                        """
                        SELECT home_rest_days, away_rest_days
                        FROM schedule_data
                        WHERE home_team_id = %s
                          AND away_team_id = %s
                          AND date          = %s
                          AND league_id     = %s
                        """,
                        (home_id, away_id, game_date, league_id)
                    )
                    if row.empty:
                        home_rest_days = None
                        away_rest_days = None
                    else:
                        home_rest_days = row["home_rest_days"].iat[0]
                        away_rest_days = row["away_rest_days"].iat[0]
                else:
                    home_rest_days = self.get_team_rest_days(home_id, game_date)
                    away_rest_days = self.get_team_rest_days(away_id, game_date)

                temp, rain = self.get_weather(home_id, game_date, game_venue_time)

                insert_sql = """
                INSERT INTO schedule_data (
                    home_team_id,
                    away_team_id,
                    date,
                    local_time,
                    venue_time,
                    league_id,
                    home_elevation_dif,
                    away_elevation_dif,
                    away_travel,
                    home_rest_days,
                    away_rest_days,
                    temperature,
                    is_raining,
                    ss_url
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    date         = VALUES(date),
                    local_time   = VALUES(local_time),
                    venue_time   = VALUES(venue_time),
                    temperature  = VALUES(temperature),
                    is_raining   = VALUES(is_raining);
                    ss_url       = IF(ss_url IS NULL, VALUES(ss_url), ss_url);
                """

                raw_params = (
                    home_id,
                    away_id,
                    game_date,
                    game_local_time,
                    game_venue_time,
                    league_id,
                    home_elevation_dif,
                    away_elevation_dif,
                    away_travel_dist,
                    home_rest_days,
                    away_rest_days,
                    temp,
                    rain,
                    ss_url
                )

                params = tuple(self._to_python(p) for p in raw_params)
                DB.execute(insert_sql, params)

            transfer_sql = """
            UPDATE  match_info AS mi
            JOIN    schedule_data AS sd
                   ON mi.home_team_id = sd.home_team_id
                  AND mi.away_team_id = sd.away_team_id
                  AND DATE(mi.date)  = sd.date
                  AND mi.league_id   = sd.league_id
            SET mi.home_elevation_dif = COALESCE(mi.home_elevation_dif, sd.home_elevation_dif),
                mi.away_elevation_dif = COALESCE(mi.away_elevation_dif, sd.away_elevation_dif),
                mi.away_travel        = COALESCE(mi.away_travel,        sd.away_travel),
                mi.home_rest_days     = COALESCE(mi.home_rest_days,     sd.home_rest_days),
                mi.away_rest_days     = COALESCE(mi.away_rest_days,     sd.away_rest_days),
                mi.temperature_c      = COALESCE(mi.temperature_c,      sd.temperature),
                mi.is_raining         = COALESCE(mi.is_raining,         sd.is_raining)
            WHERE sd.date < %s
              AND sd.league_id = %s
            """
            DB.execute(transfer_sql, (self.from_date, league_id))

            delete_sql = """
            DELETE  sd
            FROM    schedule_data AS sd
            JOIN    match_info AS mi
                   ON mi.home_team_id = sd.home_team_id
                  AND mi.away_team_id = sd.away_team_id
                  AND DATE(mi.date)  = sd.date
                  AND mi.league_id   = sd.league_id
            WHERE sd.date < %s
              AND sd.league_id = %s
            """
            DB.execute(delete_sql, (self.from_date, league_id))

            cutoff_date = self.from_date - timedelta(days=30)
            
            stale_delete_sql = """
            DELETE  sd
            FROM    schedule_data AS sd
            LEFT JOIN match_info AS mi
                   ON mi.home_team_id = sd.home_team_id
                  AND mi.away_team_id = sd.away_team_id
                  AND DATE(mi.date)  = sd.date
                  AND mi.league_id   = sd.league_id
            WHERE mi.home_team_id IS NULL
              AND sd.date < %s
              AND sd.league_id = %s
            """
            DB.execute(stale_delete_sql, (cutoff_date, league_id))

    def _schedule_exists(self, home_id, away_id, game_date, league_id):
        sql = """
        SELECT 1
        FROM schedule_data
        WHERE home_team_id = %s
          AND away_team_id = %s
          AND date          = %s
          AND league_id     = %s
        LIMIT 1
        """
        return not DB.select(sql, (home_id, away_id, game_date, league_id)).empty
    
    def _to_python(self, value):
        if value is None:
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.datetime64):
            return pd.to_datetime(value).to_pydatetime()
        return value
    
    def get_games_basic_info(self, url, upto_date):
        s = Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(url)
        driver.execute_script("window.scrollTo(0, 1000);")

        fixtures_table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.stats_table"))
        )
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")

        games_dates = []
        games_local_time = []
        games_venue_time = []
        home_teams = []
        away_teams = []

        def _parse_time(ts):
            ts = ts.strip("()")
            if not ts or ts.lower() in {"tbd", "nan"}:
                return None
            try:
                return datetime.strptime(ts, "%H:%M").time()
            except ValueError:
                return None

        for row in rows:
            date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
            date_text = date_element.text.strip()
            cleaned_date_text = re.sub(r"[^0-9-]", "", date_text)
            if not cleaned_date_text:
                continue

            game_date = datetime.strptime(cleaned_date_text, "%Y-%m-%d").date()
            if not (self.from_date <= game_date < upto_date):
                continue

            venue_time_el = row.find_element(By.CSS_SELECTOR, ".venuetime")
            local_time_el = row.find_element(By.CSS_SELECTOR, ".localtime")

            venue_time_obj = _parse_time(venue_time_el.text)
            local_time_obj = _parse_time(local_time_el.text)

            home_name = row.find_element(By.CSS_SELECTOR, "[data-stat='home_team']").text
            away_name = row.find_element(By.CSS_SELECTOR, "[data-stat='away_team']").text

            games_dates.append(game_date)
            games_local_time.append(local_time_obj)
            games_venue_time.append(venue_time_obj)
            home_teams.append(home_name)
            away_teams.append(away_name)

        driver.quit()
        return games_dates, games_local_time, games_venue_time, home_teams, away_teams

    def get_ss_urls(self, ss_url):
        s = Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(ss_url)

        try:
            popup_close = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".Button.RVwfR")))
            popup_close.click()
        except Exception as e:
            print("Popup not found or not clickable:", e)

        round_div = driver.find_elements(By.CSS_SELECTOR, ".Box.kiSsvW")

        href_list = []
        for div in round_div:
            anchor_tags = div.find_elements(By.TAG_NAME, "a")
            for a in anchor_tags:
                href = a.get_attribute("href")
                if href:
                    href_list.append(href)

        smatches_dict = {}
        for url in href_list:
            driver.get(url)
            try:
                elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".fPSBzf.iRgpoQ.bYPztT")))
                home_cont = elements[0]
                away_cont = elements[1]
                key = f"{home_cont.text} vs {away_cont.text}"
                smatches_dict[key] = url
            except:
                continue

        driver.quit()
        return smatches_dict

    def get_matched_teams_url(self, ssdict, target_title):
        match, score, _ = process.extractOne(
            target_title,
            ssdict.keys(),
            scorer=fuzz.ratio,
        )

        if score > 50:
            best_url = ssdict[match]
        else:
            best_url = None

        return best_url

    def get_team_elevation_dif(self, home_id, away_id, mode):
        teams_df = DB.select(f"SELECT * FROM team_data WHERE team_id IN ({home_id}, {away_id})")

        home_team = teams_df[teams_df["team_id"] == home_id].iloc[0]
        away_team = teams_df[teams_df["team_id"] == away_id].iloc[0]

        league_id = home_team["league_id"]

        league_df = DB.select(f"SELECT * FROM team_data WHERE league_id = {league_id}")

        league_elevation_avg = league_df["team_elevation"].mean()

        home_elevation = home_team["team_elevation"]
        away_elevation = away_team["team_elevation"]

        if mode == "home":
            reference_avg = (league_elevation_avg + home_elevation) / 2
        elif mode == "away":
            reference_avg = (league_elevation_avg + away_elevation) / 2
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'home' or 'away'.")

        elevation_difference = home_elevation - reference_avg
        return elevation_difference

    def get_travel_distance(self, home_id, away_id):
        teams_df = DB.select(f"SELECT * FROM team_data WHERE team_id IN ({home_id}, {away_id})")

        home_team = teams_df[teams_df["team_id"] == home_id].iloc[0]
        away_team = teams_df[teams_df["team_id"] == away_id].iloc[0]

        lat1, lon1 = map(str.strip, home_team["team_coordinates"].split(','))
        lat2, lon2 = map(str.strip, away_team["team_coordinates"].split(','))
    
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))

        R = 6371.0

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = int(round(R * c))
        
        return distance

    def get_team_rest_days(self, team_id, target_date): 
        team_df = DB.select(f"SELECT * FROM team_data WHERE team_id = {team_id}")

        team_fixtures_url = team_df['team_fixtures_url'].values[0]

        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(team_fixtures_url)

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="matchlogs_for"]')))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
        prev_game_date = None

        for row in rows:
            date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
            date_text = date_element.text.strip()
            cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
            if cleaned_date_text:
                game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
            else:
                continue

            if game_date < target_date:
                if prev_game_date is None or game_date > prev_game_date:
                    prev_game_date = game_date

        driver.quit()

        if prev_game_date is None:
            return 30

        rest_days = (target_date - prev_game_date).days
        return rest_days

    def get_weather(self, home_id, game_date, game_venue_time):
        if game_venue_time is None:
            return None, None

        team_df = DB.select(f"SELECT * FROM team_data WHERE team_id = {home_id}")
        lat, lon = team_df["team_coordinates"].values[0].split(",")

        today = datetime.today().date()
        base_url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            if game_date < today
            else "https://api.open-meteo.com/v1/forecast?"
        )

        dummy_date = datetime(2000, 1, 1, game_venue_time.hour, game_venue_time.minute)
        start_datetime = dummy_date - timedelta(hours=1)
        end_datetime = dummy_date + timedelta(hours=2)

        url = (
            f"{base_url}"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={game_date}&end_date={game_date}"
            f"&hourly=temperature_2m,precipitation"
            f"&timezone=auto"
        )

        response = requests.get(url)
        data = response.json()
        if "hourly" not in data:
            return None, None

        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        rains = data["hourly"]["precipitation"]

        filtered_temps = []
        filtered_rains = []
        for t, temp, rain in zip(times, temps, rains):
            dt = datetime.fromisoformat(t)
            if start_datetime.time() <= dt.time() <= end_datetime.time():
                filtered_temps.append(temp)
                filtered_rains.append(rain)

        if not filtered_temps:
            return None, None

        avg_temp = sum(filtered_temps) / len(filtered_temps)
        raining = any(r > 0.0 for r in filtered_rains)
        return avg_temp, raining

class Extract_Data:
    def __init__(self, upto_date):
        self.upto_date = upto_date
        self.get_recent_games_match_info()
        self.update_matches_info()
        self.update_pdras()
        self.update_shots()
        self.remove_old_data()

    def get_recent_games_match_info(self):
        def get_games_basic_info(url, lud):
            s=Service('chromedriver.exe')
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")  
            driver = webdriver.Chrome(service=s, options=options)
            driver.get(url)
            driver.execute_script("window.scrollTo(0, 1000);")

            fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.stats_table")))
            rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
            filtered_games_urls = []
            games_dates = []
            game_times = []
            home_teams = []
            away_teams = []
            referees = []

            for row in rows:
                date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
                date_text = date_element.text.strip()
                cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
                if cleaned_date_text:
                    game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
                else:
                    continue

                if lud <= game_date < self.upto_date:
                    games_dates.append(game_date)

                    venue_time_element = row.find_element(By.CSS_SELECTOR, '.venuetime')
                    venue_time_str = venue_time_element.text.strip("()")
                    venue_time_obj = datetime.strptime(venue_time_str, "%H:%M").time()
                    game_times.append(venue_time_obj)

                    try:
                        href_element = row.find_element(By.CSS_SELECTOR, "[data-stat='match_report'] a")
                        filtered_games_urls.append(href_element.get_attribute('href'))
                    except NoSuchElementException:
                        continue

                    home_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='home_team']")
                    home_name = home_name_element.text
                    home_teams.append(home_name)

                    away_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='away_team']")
                    away_name = away_name_element.text
                    away_teams.append(away_name)

                    referee_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='referee']")
                    referee_name = referee_name_element.text           
                    referees.append(referee_name)
            driver.quit()
            
            return filtered_games_urls, games_dates, game_times, home_teams, away_teams, referees
    
        active_leagues_df = DB.select("SELECT * FROM league_data WHERE is_active = 1")

        insert_sql = """
        INSERT IGNORE INTO match_info (
            home_team_id, away_team_id, date, league_id, referee_name, url
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        update_sql = "UPDATE league_data SET last_updated_date = %s WHERE league_id = %s"
        
        for league_id in tqdm(active_leagues_df["league_id"].tolist()):
            url = active_leagues_df[active_leagues_df['league_id'] == league_id]['fbref_fixtures_url'].values[0]
            lud = active_leagues_df[active_leagues_df['league_id'] == league_id]['last_updated_date'].values[0]
            games_url, games_dates, game_times, home_teams, away_teams, referees = get_games_basic_info(url, lud)

            for i in tqdm(range(len(games_url))):
                game_url = games_url[i]
                game_date = games_dates[i]
                game_time = game_times[i]
                game_datetime = datetime.combine(game_date, game_time)
                home_team = home_teams[i]
                home_id = get_team_id_by_name(home_team)
                away_team = away_teams[i]
                away_id = get_team_id_by_name(away_team)
                referee = referees[i]

                params = (
                    home_id,
                    away_id,
                    game_datetime,
                    league_id,
                    referee,
                    game_url,
                )
                DB.execute(insert_sql, params)

            DB.execute(update_sql, (self.upto_date, league_id))

    def update_matches_info(self):
        """
        Get the matches_id that are missing to update them for the breakdown and detail tables.
        """
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

        match_info = DB.select("SELECT match_id, url, home_team_id, away_team_id FROM match_info")
        match_detail = DB.select("SELECT match_id FROM match_detail")
        match_breakdown = DB.select("SELECT match_id FROM match_breakdown")

        detail_ids = set(match_detail['match_id'])
        breakdown_ids = set(match_breakdown['match_id'])

        existing_in_both = detail_ids.intersection(breakdown_ids)

        missing_matches_df = match_info[~match_info['match_id'].isin(existing_in_both)]

        for _, mi_row in tqdm(missing_matches_df.iterrows(), total=len(missing_matches_df)):
            # Match detail
            s = Service('chromedriver.exe')
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")  
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--blink-settings=imagesEnabled=false")
            options.add_argument("--ignore-certificate-errors")
            driver = webdriver.Chrome(service=s, options=options)
            driver.get(mi_row['url'])
            home_team       = get_team_name_by_id(mi_row['home_team_id'])
            away_team       = get_team_name_by_id(mi_row['away_team_id'])
            home_team_id    = mi_row['home_team_id']
            away_team_id    = mi_row['away_team_id']
            match_id        = mi_row["match_id"]

            home_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="a"]/table')))
            home_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", home_table))

            away_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="b"]/table')))
            away_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", away_table))

            home_players = extract_players(home_data, home_team)
            away_players = extract_players(away_data, away_team)

            try:
                events_wrap = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="events_wrap"]')))
            except:
                return None

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

            try:
                shots_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="shots_all"]')))
            except TimeoutException:
                driver.quit()
                DB.execute("DELETE FROM match_info WHERE match_id = %s", (mi_row["match_id"],))
                continue
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
                for _, seg_row in seg_shots.iterrows():
                    if home_team in seg_row['Squad']:
                        if "Head" in seg_row['Body Part']:
                            teamA_headers += 1
                            teamA_hxg += seg_row['xG']
                        elif "Foot" in seg_row['Body Part']:
                            teamA_footers += 1
                            teamA_fxg += seg_row['xG']
                    elif away_team in seg_row['Squad']:
                        if "Head" in seg_row['Body Part']:
                            teamB_headers += 1
                            teamB_hxg += seg_row['xG']
                        elif "Foot" in seg_row['Body Part']:
                            teamB_footers += 1
                            teamB_fxg += seg_row['xG']

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
                red_diff = cum_red_away - cum_red_home
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
                DB.execute(sql, params)

            # Match breakdown
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
                        new_stats = initialize_player_stats([player_in])
                        new_stats[player_in]["starter"] = False
                        new_stats[player_in]["sub_in_min"] = sub_minute
                        new_stats[player_in]["in_status"] = state
                        home_player_stats.update(new_stats)
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
                        new_stats = initialize_player_stats([player_in])
                        new_stats[player_in]["starter"] = False
                        new_stats[player_in]["sub_in_min"] = sub_minute
                        new_stats[player_in]["in_status"] = state
                        away_player_stats.update(new_stats)
            
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

            all_shots = shots_df.loc[:, [
                ('Unnamed: 0_level_0', 'Minute'),
                ('Unnamed: 1_level_0', 'Player'),
                ('Unnamed: 2_level_0', 'Squad'),
                ('Unnamed: 3_level_0', 'xG'),
                ('Unnamed: 4_level_0', 'PSxG'),
                ('Unnamed: 5_level_0', 'Outcome'),
                ('Unnamed: 7_level_0', 'Body Part'),
                ('SCA 1', 'Player'),
                ('SCA 1', 'Event')
            ]]

            cleaned_columns = []
            for col in all_shots.columns:
                if 'Unnamed' in col[0]:
                    cleaned_columns.append(col[1])
                else:
                    cleaned_columns.append('_'.join(col).strip())

            all_shots.columns = cleaned_columns

            all_shots = all_shots[all_shots['Minute'].notna() & (all_shots['Minute'] != 'Minute')]

            all_shots['Minute'] = all_shots['Minute'].astype(str).str.replace(r'\+.*', '', regex=True).str.strip().astype(float).astype(int)
            all_shots['xG'] = all_shots['xG'].astype(float)
            all_shots['PSxG'] = all_shots['PSxG'].astype(float)
            all_shots['Body Part'] = all_shots['Body Part'].astype(str).str.strip()
            all_shots['Player'] = all_shots['Player'].astype(str).str.strip()
            all_shots['SCA 1_Player'] = all_shots['SCA 1_Player'].astype(str).str.strip()
            all_shots['SCA 1_Event'] = all_shots['SCA 1_Event'].astype(str).str.strip()

            for idx, shot_row in all_shots.iterrows():
                shooter_name = shot_row["Player"].strip()
                shot_body = shot_row["Body Part"]
                shot_xg = float(shot_row["xG"])
                shot_psxg = float(shot_row["PSxG"])
                outcome = shot_row["Outcome"]
                sca_event = shot_row["SCA 1_Event"]
                sca_player = shot_row["SCA 1_Player"].strip()

                if math.isnan(shot_xg):
                    shot_xg = 0.00

                if math.isnan(shot_psxg):
                    shot_psxg = 0.00
                
                shooter_team_stats = None
                opponent_gk_stats = None
                shooter_key = None
                for key in home_player_stats:
                    if key.split("_")[0] == shooter_name:
                        shooter_team_stats = home_player_stats
                        shooter_key = key
                        opponent_gk_stats = away_player_stats
                        break
                if shooter_key is None:
                    for key in away_player_stats:
                        if key.split("_")[0] == shooter_name:
                            shooter_team_stats = away_player_stats
                            shooter_key = key
                            opponent_gk_stats = home_player_stats
                            break
                if shooter_key is None:
                    continue
                
                shooter_team = "home" if shooter_key in home_player_stats else "away"

                shot_type = None
                if "Head" in shot_body:
                    shot_type = "head"
                elif "Foot" in shot_body:
                    shot_type = "foot"
                else:
                    continue
                
                if shot_type == "head":
                    shooter_team_stats[shooter_key]["headers"] += 1
                    shooter_team_stats[shooter_key]["hxg"] += shot_xg
                    shooter_team_stats[shooter_key]["hpsxg"] += shot_psxg
                elif shot_type == "foot":
                    shooter_team_stats[shooter_key]["footers"] += 1
                    shooter_team_stats[shooter_key]["fxg"] += shot_xg
                    shooter_team_stats[shooter_key]["fpsxg"] += shot_psxg
                
                if "Pass" in sca_event:
                    assist_key = None
                    for key in shooter_team_stats:
                        if key.split("_")[0] == sca_player:
                            assist_key = key
                            break
                    if assist_key:
                        shooter_team_stats[assist_key]["key_passes"] += 1
                        if shot_type == "head":
                            shooter_team_stats[assist_key]["kp_hxg"] += shot_xg
                        elif shot_type == "foot":
                            shooter_team_stats[assist_key]["kp_fxg"] += shot_xg
                else:
                    shooter_team_stats[shooter_key]["non_assisted_footers"] += 1
                
                opponent_gk_key = list(opponent_gk_stats.keys())[0]
                opponent_gk_stats[opponent_gk_key]["gk_psxg"] += shot_psxg
                if outcome == "Goal":
                    opponent_gk_stats[opponent_gk_key]["gk_ga"] += 1

                # Shots data
                shot_minute = int(shot_row["Minute"])

                if shooter_key in home_player_stats:
                    shooter_team = "home"
                else:
                    shooter_team = "away"

                if shooter_team == "home":
                    off_players = get_lineups(home_players, subs_events, shot_minute, "home")
                    def_players = get_lineups(away_players, subs_events, shot_minute, "away")
                else:
                    off_players = get_lineups(away_players, subs_events, shot_minute, "away")
                    def_players = get_lineups(home_players, subs_events, shot_minute, "home")

                cum_goal_home = sum(1 for minute, t in goal_events if minute <= shot_minute and t == "home")
                cum_goal_away = sum(1 for minute, t in goal_events if minute <= shot_minute and t == "away")

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

                red_diff = cum_red_home - cum_red_away
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

                shooter_id = shooter_team_stats[shooter_key]["player_id"]
                if "Pass" in sca_event and 'assist_key' in locals() and assist_key:
                    assister_id = shooter_team_stats[assist_key]["player_id"]
                else:
                    assister_id = ""

                GK_id = opponent_gk_stats[list(opponent_gk_stats.keys())[0]]["player_id"]

                team_id = home_team_id if shooter_team == "home" else away_team_id

                sql_shot = "INSERT IGNORE INTO shots_data (match_id, xg, psxg, outcome, shooter_id, assister_id, team_id, GK_id, off_players, def_players, match_state, player_dif, shot_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                params_shot = (match_id,
                               shot_xg,
                               shot_psxg,
                               1 if outcome == "Goal" else 0,
                               shooter_id,
                               assister_id,
                               team_id,
                               GK_id,
                               json.dumps(off_players, ensure_ascii=False),
                               json.dumps(def_players, ensure_ascii=False),
                               match_state,
                               player_dif,
                               shot_type)
                DB.execute(sql_shot, params_shot)

            tabbed_tables = driver.find_elements(By.CSS_SELECTOR, ".table_wrapper.tabbed")

            for table in tabbed_tables:
                heading = table.find_element(By.CLASS_NAME, "section_heading").text
                clean_heading = heading[:re.search(r'\bPlayer Stats\b', heading).start()].strip() if re.search(r'\bPlayer Stats\b', heading) else heading.strip()
                team_initials = ''.join(word[0].upper() for word in clean_heading.split() if word)
                try:
                    switcher = table.find_element(By.CSS_SELECTOR, ".filter.switcher")
                    tabs = switcher.find_elements(By.TAG_NAME, "a")
                    for tab in tabs:
                        if "Miscellaneous Stats" in tab.text:
                            driver.execute_script("arguments[0].click();", tab)

                            active_container = WebDriverWait(table, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".table_container.tabbed.is_setup.current")))
                            
                            stats_table = active_container.find_element(By.CSS_SELECTOR, ".stats_table.sortable.now_sortable")
                            tbody = stats_table.find_element(By.TAG_NAME, "tbody")
                            rows = tbody.find_elements(By.TAG_NAME, "tr")

                            for row in rows:
                                player_name = row.find_element(By.CSS_SELECTOR, '[data-stat="player"]').text.strip()
                                shirt_number = row.find_element(By.CSS_SELECTOR, '[data-stat="shirtnumber"]').text.strip()
                                fouls_committed_text = row.find_element(By.CSS_SELECTOR, '[data-stat="fouls"]').text.strip()
                                fouls_drawn_text = row.find_element(By.CSS_SELECTOR, '[data-stat="fouled"]').text.strip()
                                yellow_text = row.find_element(By.CSS_SELECTOR, '[data-stat="cards_yellow"]').text.strip()
                                red_text = row.find_element(By.CSS_SELECTOR, '[data-stat="cards_red"]').text.strip()
                                fouls_committed_val = int(fouls_committed_text) if fouls_committed_text.isdigit() else 0
                                fouls_drawn_val = int(fouls_drawn_text) if fouls_drawn_text.isdigit() else 0
                                yellow_val = int(yellow_text) if yellow_text.isdigit() else 0
                                red_val = int(red_text) if red_text.isdigit() else 0
                                player_key = f"{player_name}_{shirt_number}_{team_initials}"
                                if player_key in home_player_stats:
                                    home_player_stats[player_key]["fouls_committed"] = fouls_committed_val
                                    home_player_stats[player_key]["fouls_drawn"] = fouls_drawn_val
                                    home_player_stats[player_key]["yellow_cards"] = yellow_val
                                    home_player_stats[player_key]["red_cards"] = red_val
                                elif player_key in away_player_stats:
                                    away_player_stats[player_key]["fouls_committed"] = fouls_committed_val
                                    away_player_stats[player_key]["fouls_drawn"] = fouls_drawn_val
                                    away_player_stats[player_key]["yellow_cards"] = yellow_val
                                    away_player_stats[player_key]["red_cards"] = red_val
                            break
                except Exception as e:
                    print(f"Error processing table: {e}")

            insert_sql = "INSERT IGNORE INTO match_breakdown (match_id, player_id, headers, footers, key_passes, non_assisted_footers, hxg, fxg, kp_hxg, kp_fxg, hpsxg, fpsxg, gk_psxg, gk_ga, sub_in, sub_out, in_status, out_status, fouls_committed, fouls_drawn, yellow_cards, red_cards, minutes_played) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

            for team_stats in (home_player_stats, away_player_stats):
                for player, stat in team_stats.items():
                    if stat.get("sub_out_min", 0) == 0:
                        continue
                    params = (match_id,
                            stat["player_id"],
                            stat["headers"],
                            stat["footers"],
                            stat["key_passes"],
                            stat["non_assisted_footers"],
                            stat["hxg"],
                            stat["fxg"],
                            stat["kp_hxg"],
                            stat["kp_fxg"],
                            stat["hpsxg"],
                            stat["fpsxg"],
                            stat["gk_psxg"],
                            stat["gk_ga"],
                            stat["sub_in_min"],
                            stat["sub_out_min"],
                            stat["in_status"],
                            stat["out_status"],
                            stat.get("fouls_committed", 0),
                            stat.get("fouls_drawn", 0),
                            stat.get("yellow_cards", 0),
                            stat.get("red_cards", 0),
                            stat["minutes_played"])
                    DB.execute(insert_sql, params)

            driver.quit()
            
    def update_pdras(self):
        """
        Before processing new data, update the pre defined RAS for old matches.
        """
        non_pdras_matches_df = DB.select("""
            SELECT 
                md.detail_id,
                md.match_id,
                mi.league_id,
                md.teamA_players,
                md.teamB_players,
                md.minutes_played
            FROM match_detail md
            JOIN match_info mi ON mi.match_id = md.match_id
            WHERE md.teamA_pdras IS NULL 
               OR md.teamB_pdras IS NULL;
        """)

        non_pdras_matches_df['teamA_players'] = non_pdras_matches_df['teamA_players'].apply(
            lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        )
        non_pdras_matches_df['teamB_players'] = non_pdras_matches_df['teamB_players'].apply(
            lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        )

        players_needed = set()
        for _, row in non_pdras_matches_df.iterrows():
            players_needed.update(row['teamA_players'])
            players_needed.update(row['teamB_players'])

        if players_needed:
            placeholders = ','.join(['%s'] * len(players_needed))
            players_sql = (
                f"SELECT player_id, off_sh_coef, def_sh_coef "
                f"FROM players_data "
                f"WHERE player_id IN ({placeholders});"
            )
            players_coef_df = DB.select(players_sql, list(players_needed))
            off_sh_coef_dict = players_coef_df.set_index("player_id")["off_sh_coef"].to_dict()
            def_sh_coef_dict = players_coef_df.set_index("player_id")["def_sh_coef"].to_dict()
        else:
            off_sh_coef_dict, def_sh_coef_dict = {}, {}

        baseline_df = DB.select("SELECT league_id, sh_baseline_coef FROM league_data;")
        baseline_dict = baseline_df.set_index("league_id")["sh_baseline_coef"].fillna(0).to_dict()

        for _, row in non_pdras_matches_df.iterrows():
            minutes = row['minutes_played']
            league_id = row['league_id']
            baseline = baseline_dict.get(league_id, 0.0)

            teamA_ids = row['teamA_players']
            teamB_ids = row['teamB_players']

            teamA_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_pdras = (baseline + teamA_offense - teamB_defense) * minutes

            teamB_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_pdras = (baseline + teamB_offense - teamA_defense) * minutes

            DB.execute(
                "UPDATE match_detail SET teamA_pdras = %s, teamB_pdras = %s WHERE detail_id = %s",
                (teamA_pdras, teamB_pdras, row['detail_id'])
            )

    def update_shots(self):
        """
        Before processing new data, update the shots plsqa, shooter and assister sq. And then the rsq, shooter  and gk ability.
        """
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
            
            df = DB.select(sql)
            
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
                eval_metric      = "logloss",
                tree_method      = "hist",
                max_depth        = 6,
                eta              = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                min_child_weight = 2
            )

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=500,
                nfold=5,
                early_stopping_rounds=100,
                metrics='logloss',
                verbose_eval=False
            )
            
            optimal_rounds = len(cv_results)
            booster = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
            
            return booster, X.columns.tolist()

        def predict_refined_sq(booster: xgb.Booster,
                            feature_columns: list[str],
                            shot_features: dict,
                            *,
                            raw: bool = False) -> float:
            
            cat_cols = ['match_state', 'player_dif']
            num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']
            
            row = shot_features.copy()
            
            for c in cat_cols:
                v = row.get(c)
                row[c] = np.nan if pd.isna(v) else str(v)
            
            num_df = pd.DataFrame([{k: row[k] for k in num_cols}])
            cat_df = pd.get_dummies(
                pd.DataFrame([{c: row[c] for c in cat_cols}]),
                prefix       = cat_cols,
                dummy_na     = True
            )
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

        non_updated_shots_df = DB.select("SELECT * FROM shots_data;")

        team_ids = non_updated_shots_df["team_id"].dropna().unique().tolist()

        if team_ids:
            placeholders  = ",".join(["%s"] * len(team_ids))
            team_sql      = f"""
                SELECT team_id, league_id
                FROM team_data
                WHERE team_id IN ({placeholders});
            """
            team_df            = DB.select(team_sql, team_ids)
            team_to_league_map = dict(zip(team_df["team_id"], team_df["league_id"]))
            league_ids         = list({v for v in team_to_league_map.values() if v is not None})
        else:
            team_to_league_map = {}
            league_ids         = []

        non_updated_shots_df["league_id"] = non_updated_shots_df["team_id"].map(team_to_league_map)

        if league_ids:
            placeholders = ",".join(["%s"] * len(league_ids))
            baseline_sql = f"""
                SELECT league_id,
                       hxg_baseline_coef,
                       fxg_baseline_coef
                FROM league_data
                WHERE league_id IN ({placeholders});
            """
            baseline_df   = DB.select(baseline_sql, league_ids)
            baseline_dict = {
                r["league_id"]: {
                    "hxg": float(r["hxg_baseline_coef"] or 0.0),
                    "fxg": float(r["fxg_baseline_coef"] or 0.0)
                }
                for _, r in baseline_df.iterrows()
            }
        else:
            baseline_dict = {}

        non_updated_shots_df['off_players'] = non_updated_shots_df['off_players'].apply(
            lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        )
        non_updated_shots_df['def_players'] = non_updated_shots_df['def_players'].apply(
            lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        )

        players_needed = set()
        for _, row in non_updated_shots_df.iterrows():
            players_needed.update(row['off_players'])
            players_needed.update(row['def_players'])
            if row['assister_id']:
                players_needed.add(row['assister_id'])
            players_needed.add(row['shooter_id'])
            players_needed.add(row['GK_id'])

        if players_needed:
            placeholders = ','.join(['%s'] * len(players_needed))
            players_sql = (
                f"SELECT player_id, off_hxg_coef, def_hxg_coef, off_fxg_coef, def_fxg_coef, headers, footers, key_passes, hxg, fxg, kp_hxg, kp_fxg, hpsxg, fpsxg, gk_psxg, gk_ga "
                f"FROM players_data "
                f"WHERE player_id IN ({placeholders});"
            )
            players_data_df  = DB.select(players_sql, list(players_needed))
            p_dict = players_data_df.set_index("player_id").to_dict("index")
        else:
            p_dict = {}

        for _, row in tqdm(non_updated_shots_df.iterrows(), total=len(non_updated_shots_df)):
            league_id   = row["league_id"]
            hxg_base    = baseline_dict.get(league_id, {}).get("hxg", 0.0)
            fxg_base    = baseline_dict.get(league_id, {}).get("fxg", 0.0)

            off_ids = row['off_players']
            def_ids = row['def_players']
            bp = row['shot_type']
            shooter_id = row['shooter_id']
            assister_id = row['assister_id']
            gk_id = row['GK_id']

            if bp == "head":
                offense = sum(p_dict.get(pid, {}).get('off_hxg_coef', 0) for pid in off_ids)
                defense = sum(p_dict.get(pid, {}).get('def_hxg_coef', 0) for pid in def_ids)
                plsqa   = hxg_base + offense - defense
            else:
                offense = sum(p_dict.get(pid, {}).get('off_fxg_coef', 0) for pid in off_ids)
                defense = sum(p_dict.get(pid, {}).get('def_fxg_coef', 0) for pid in def_ids)
                plsqa   = fxg_base + offense - defense

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

            if not assister_id or assister_id not in p_dict:
                assister_sq = None
            else:
                assister_data = p_dict[assister_id]
                if bp == "head":
                    num = assister_data.get('kp_hxg', 0)
                else:
                    num = assister_data.get('kp_fxg', 0)
                den         = assister_data.get('key_passes', 1)
                assister_sq = num / den if den else 0.0

            gk_data = p_dict.get(gk_id, {})
            gk_psxg = gk_data.get('gk_psxg', 0)
            gk_ga   = gk_data.get('gk_ga',   0)
            gk_A    = 1.0 - (gk_ga / gk_psxg) if gk_psxg else 0.0

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

            DB.execute(
                "UPDATE shots_data SET total_PLSQA = %s, shooter_SQ = %s, assister_SQ = %s, RSQ = %s, shooter_A = %s, GK_A = %s WHERE shot_id = %s",
                (plsqa, shooter_sq, assister_sq, rsq, shooter_A, gk_A, row['shot_id'])
            )

    def remove_old_data(self):
        self.a_year_ago_date = self.upto_date - timedelta(days=365)

        delete_query  = """
        DELETE FROM match_info 
        WHERE date < %s
        """

        DB.execute(delete_query, (self.a_year_ago_date,))

# ------------------------------ Process data ------------------------------
class Process_Data:
    def __init__(self):
        """
        Class to reset the players_data table and fill it with new data.
        """

        # DB.execute("TRUNCATE TABLE players_data;")
        # DB.execute("TRUNCATE TABLE referee_data;")

        # self.insert_players_basics()
        # self.update_players_shots_coef()
        # self.update_players_totals()
        # self.update_players_xg_coef()
        # self.update_match_info_referee_totals()
        # self.update_referee_data_totals()
        self.train_context_ras_model()
        # self.train_refined_sq_model()
        # self.train_post_shot_goal_model()

    def insert_players_basics(self):
        """
        Function to insert basic information from all players into players_data from match detail without duplicating.
        """
        sql = """
        SELECT md.teamA_players, md.teamB_players, mi.home_team_id, mi.away_team_id 
        FROM match_detail md 
        JOIN match_info mi ON md.match_id = mi.match_id 
        """
        result = DB.select(sql, ())
        
        if result.empty:
            return 0

        players_set = set()
        for _, row in result.iterrows():
            teamA_players = json.loads(row["teamA_players"])
            teamB_players = json.loads(row["teamB_players"])
            home_team = int(row["home_team_id"])
            away_team = int(row["away_team_id"])
        
            for player in teamA_players:
                players_set.add((player, home_team))
            
            for player in teamB_players:
                players_set.add((player, away_team))
        
        insert_sql = "INSERT IGNORE INTO players_data (player_id, current_team) VALUES (%s, %s)"
        DB.execute(insert_sql, list(players_set), many=True)

    def update_players_shots_coef(self):
        """
        Function to update players shot types coefficients per league
        """
        league_id_df = DB.select("SELECT league_id FROM league_data WHERE is_active = 1")

        for league_id in league_id_df['league_id'].tolist():
            baseline_total = 0.0  
            for shot_type in ["headers", "footers"]:
                league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE league_id = {league_id}")
                matches_ids = league_matches_df['match_id'].tolist()

                if not matches_ids:
                    continue            

                matches_ids_placeholder = ','.join(['%s'] * len(matches_ids))
                matches_sql = f"""
                SELECT 
                    teamA_players, 
                    teamB_players, 
                    teamA_{shot_type}, 
                    teamB_{shot_type}, 
                    minutes_played 
                FROM match_detail 
                WHERE match_id IN ({matches_ids_placeholder});
                """
                matches_details_df = DB.select(matches_sql, matches_ids)

                matches_details_df['teamA_players'] = matches_details_df['teamA_players'].apply(
                    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
                )
                matches_details_df['teamB_players'] = matches_details_df['teamB_players'].apply(
                    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
                )

                players_set = set()
                for idx, row in matches_details_df.iterrows():
                    players_set.update(row['teamA_players'])
                    players_set.update(row['teamB_players'])
                players = sorted(list(players_set))
                num_players = len(players)
                players_to_index = {player: idx for idx, player in enumerate(players)}

                rows = []
                cols = []
                data_vals = []
                y = []
                sample_weights = []
                row_num = 0

                for idx, row in matches_details_df.iterrows():
                    minutes = row['minutes_played']
                    if minutes == 0:
                        continue
                    teamA_players = row['teamA_players']
                    teamB_players = row['teamB_players']
                    teamA_st = row[f'teamA_{shot_type}']
                    teamB_st = row[f'teamB_{shot_type}']

                    for p in teamA_players:
                        rows.append(row_num)
                        cols.append(players_to_index[p])
                        data_vals.append(1)
                    for p in teamB_players:
                        rows.append(row_num)
                        cols.append(num_players + players_to_index[p])
                        data_vals.append(-1)
                    y.append(teamA_st / minutes)
                    sample_weights.append(minutes)
                    row_num += 1

                    for p in teamB_players:
                        rows.append(row_num)
                        cols.append(players_to_index[p])
                        data_vals.append(1)
                    for p in teamA_players:
                        rows.append(row_num)
                        cols.append(num_players + players_to_index[p])
                        data_vals.append(-1)
                    y.append(teamB_st / minutes)
                    sample_weights.append(minutes)
                    row_num += 1

                alphas = np.logspace(-3, 3, 13)

                X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
                y_array = np.array(y)
                sample_weights_array = np.array(sample_weights)

                ridge_base = Ridge(fit_intercept=True, solver='sparse_cg')
                grid = GridSearchCV(ridge_base,
                                    param_grid={'alpha': alphas},
                                    scoring='neg_mean_squared_error',
                                    cv=5,
                                    n_jobs=-1)
                grid.fit(X, y_array, sample_weight=sample_weights_array)

                best_model = grid.best_estimator_
                baseline_total += float(best_model.intercept_)

                offensive_ratings = dict(zip(players, best_model.coef_[:num_players]))
                defensive_ratings = dict(zip(players, best_model.coef_[num_players:]))

                for player in players:
                    off_sh = offensive_ratings[player]
                    def_sh = defensive_ratings[player]
                    update_coef_query = f"""
                    UPDATE players_data
                    SET off_{shot_type}_coef = %s, def_{shot_type}_coef = %s
                    WHERE player_id = %s
                    """
                    DB.execute(update_coef_query, (float(off_sh), float(def_sh), player))
            DB.execute("""
                UPDATE league_data
                SET sh_baseline_coef = %s
                WHERE league_id = %s
            """, (baseline_total, league_id))
        sum_coef_sql = """
        UPDATE players_data
        SET off_sh_coef = COALESCE(off_headers_coef, 0) + COALESCE(off_footers_coef, 0),
            def_sh_coef = COALESCE(def_headers_coef, 0) + COALESCE(def_footers_coef, 0)
        """
        DB.execute(sum_coef_sql)

    def update_players_totals(self):
        """
        Function to sum all information from all players (& referee) into players_data (referee_data) from match breakdown (match_info).
        """
        players_id_df = DB.select("SELECT DISTINCT player_id FROM players_data")
        
        for player_id in players_id_df["player_id"].tolist():
            pagg_query = """
            SELECT
                COALESCE(SUM(headers), 0) AS headers,
                COALESCE(SUM(footers), 0) AS footers,
                COALESCE(SUM(key_passes), 0) AS key_passes,
                COALESCE(SUM(non_assisted_footers), 0) AS non_assisted_footers,
                COALESCE(SUM(minutes_played), 0) AS minutes_played,
                COALESCE(SUM(hxg), 0) AS hxg,
                COALESCE(SUM(fxg), 0) AS fxg,
                COALESCE(SUM(kp_hxg), 0) AS kp_hxg,
                COALESCE(SUM(kp_fxg), 0) AS kp_fxg,
                COALESCE(SUM(hpsxg), 0) AS hpsxg,
                COALESCE(SUM(fpsxg), 0) AS fpsxg,
                COALESCE(SUM(gk_psxg), 0) AS gk_psxg,
                COALESCE(SUM(gk_ga), 0) AS gk_ga,
                COALESCE(SUM(fouls_committed), 0) AS fouls_committed,
                COALESCE(SUM(fouls_drawn), 0) AS fouls_drawn,
                COALESCE(SUM(yellow_cards), 0) AS yellow_cards,
                COALESCE(SUM(red_cards), 0) AS red_cards
            FROM match_breakdown
            WHERE player_id = %s
            """
            pagg_result = DB.select(pagg_query, (player_id,))
            if pagg_result.empty:
                continue

            row = pagg_result.iloc[0]

            status_query = """
            SELECT in_status, out_status, sub_in, sub_out
            FROM match_breakdown
            WHERE player_id = %s
            """
            status_result = DB.select(status_query, (player_id,))
            
            in_status_dict = {"trailing": 0, "level": 0, "leading": 0}
            out_status_dict = {"trailing": 0, "level": 0, "leading": 0}
            subs_in_list = []
            subs_out_list = []
            
            for _, status_row in status_result.iterrows():
                in_stat = status_row["in_status"]
                out_stat = status_row["out_status"]

                if in_stat in in_status_dict:
                    in_status_dict[in_stat] += 1
                if out_stat in out_status_dict:
                    out_status_dict[out_stat] += 1

                sub_in_val = status_row["sub_in"]
                if pd.notna(sub_in_val) and sub_in_val != "":
                    subs_in_list.append(sub_in_val)

                sub_out_val = status_row["sub_out"]
                if pd.notna(sub_out_val) and sub_out_val != "":
                    subs_out_list.append(sub_out_val)

            pupdate_query = """
            UPDATE players_data
            SET 
                headers = %s,
                footers = %s,
                key_passes = %s,
                non_assisted_footers = %s,
                minutes_played = %s,
                hxg = %s,
                fxg = %s,
                kp_hxg = %s,
                kp_fxg = %s,
                hpsxg = %s,
                fpsxg = %s,
                gk_psxg = %s,
                gk_ga = %s,
                fouls_committed = %s,
                fouls_drawn = %s,
                yellow_cards = %s,
                red_cards = %s,
                in_status = %s,
                out_status = %s,
                sub_in = %s,
                sub_out = %s
            WHERE player_id = %s
            """
            DB.execute(pupdate_query, (
                row["headers"],
                row["footers"],
                row["key_passes"],
                row["non_assisted_footers"],
                row["minutes_played"],
                row["hxg"],
                row["fxg"],
                row["kp_hxg"],
                row["kp_fxg"],
                row["hpsxg"],
                row["fpsxg"],
                row["gk_psxg"],
                row["gk_ga"],
                row["fouls_committed"],
                row["fouls_drawn"],
                row["yellow_cards"],
                row["red_cards"],
                json.dumps(in_status_dict),
                json.dumps(out_status_dict),
                json.dumps(subs_in_list, allow_nan=False),
                json.dumps(subs_out_list, allow_nan=False),
                player_id
            ))

    def update_players_xg_coef(self):
        """
        Function to update players xg coefficients per league
        """
        league_id_df = DB.select("SELECT league_id FROM league_data WHERE is_active = 1")
        
        for league_id in league_id_df['league_id'].tolist():
            baseline_per_type = {"headers": 0.0, "footers": 0.0}
            for shot_type in ["headers", "footers"]:
                prefix = "h" if shot_type == "headers" else "f"
                league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE league_id = {league_id}")
                matches_ids = league_matches_df['match_id'].tolist()
                if not matches_ids:
                    continue
                matches_ids_placeholder = ','.join(['%s'] * len(matches_ids))
                matches_sql = f"""
                SELECT 
                    teamA_players, 
                    teamB_players, 
                    teamA_{shot_type}, 
                    teamB_{shot_type},
                    teamA_{prefix}xg as teamA_xg,
                    teamB_{prefix}xg as teamB_xg
                FROM match_detail 
                WHERE match_id IN ({matches_ids_placeholder});
                """
                matches_details_df = DB.select(matches_sql, matches_ids)

                matches_details_df['teamA_players'] = matches_details_df['teamA_players'].apply(
                    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
                )
                matches_details_df['teamB_players'] = matches_details_df['teamB_players'].apply(
                    lambda v: v if isinstance(v, list) else ast.literal_eval(v)
                )
                
                players_set = set()
                for _, row in matches_details_df.iterrows():
                    players_set.update(row['teamA_players'])
                    players_set.update(row['teamB_players'])
                players = sorted(list(players_set))
                num_players = len(players)
                players_to_index = {player: idx for idx, player in enumerate(players)}
                
                rows, cols, data_vals, y, sample_weights = [], [], [], [], []
                row_num = 0
                
                for _, row in matches_details_df.iterrows():
                    shots_teamA = row[f'teamA_{shot_type}']
                    shots_teamB = row[f'teamB_{shot_type}']
                    
                    if shots_teamA > 0:
                        xg_teamA = row['teamA_xg']
                        for p in row['teamA_players']:
                            rows.append(row_num)
                            cols.append(players_to_index[p])
                            data_vals.append(1)
                        for p in row['teamB_players']:
                            rows.append(row_num)
                            cols.append(num_players + players_to_index[p])
                            data_vals.append(-1)
                        y.append(xg_teamA / shots_teamA)
                        sample_weights.append(shots_teamA)
                        row_num += 1
                        
                    if shots_teamB > 0:
                        xg_teamB = row['teamB_xg']
                        for p in row['teamB_players']:
                            rows.append(row_num)
                            cols.append(players_to_index[p])
                            data_vals.append(1)
                        for p in row['teamA_players']:
                            rows.append(row_num)
                            cols.append(num_players + players_to_index[p])
                            data_vals.append(-1)
                        y.append(xg_teamB / shots_teamB)
                        sample_weights.append(shots_teamB)
                        row_num += 1
                
                if row_num == 0:
                    continue
                
                X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
                y_array = np.array(y)
                sample_weights_array = np.array(sample_weights)

                alphas = np.logspace(-3, 3, 13)
                
                ridge_base = Ridge(fit_intercept=True, solver='sparse_cg')
                grid = GridSearchCV(
                    ridge_base,
                    param_grid={'alpha': alphas},
                    scoring='neg_mean_squared_error',
                    cv=5,
                    n_jobs=-1
                )
                grid.fit(X, y_array, sample_weight=sample_weights_array)
                best_model = grid.best_estimator_
                baseline_per_type[shot_type] += float(best_model.intercept_)
                
                offensive_ratings = dict(zip(players, best_model.coef_[:num_players]))
                defensive_ratings = dict(zip(players, best_model.coef_[num_players:]))
                
                for player in players:
                    off_coef = offensive_ratings[player]
                    def_coef = defensive_ratings[player]
                    
                    if shot_type == "headers":
                        update_coef_query = """
                        UPDATE players_data
                        SET off_hxg_coef = %s, def_hxg_coef = %s
                        WHERE player_id = %s
                        """
                    else:
                        update_coef_query = """
                        UPDATE players_data
                        SET off_fxg_coef = %s, def_fxg_coef = %s
                        WHERE player_id = %s
                        """
                    
                    DB.execute(update_coef_query, (off_coef, def_coef, player))
            DB.execute("""
                UPDATE league_data
                SET hxg_baseline_coef = %s, fxg_baseline_coef = %s
                WHERE league_id = %s
            """, (baseline_per_type["headers"], baseline_per_type["footers"], league_id))

    def update_match_info_referee_totals(self):
        sql = """
        UPDATE match_info AS mi
        JOIN (
            SELECT  match_id,
                    COALESCE(SUM(fouls_committed),0) AS total_fouls,
                    COALESCE(SUM(yellow_cards),0)    AS yellow_cards,
                    COALESCE(SUM(red_cards),0)       AS red_cards
            FROM    match_breakdown
            GROUP BY match_id
        ) AS mb ON mb.match_id = mi.match_id
        SET mi.total_fouls  = mb.total_fouls,
            mi.yellow_cards = mb.yellow_cards,
            mi.red_cards    = mb.red_cards
        WHERE mi.total_fouls = 0;
        """
        DB.execute(sql)

    def update_referee_data_totals(self):
        sql = """
        INSERT INTO referee_data
                (referee_name, fouls, yellow_cards, red_cards, matches_played)

        SELECT  referee_name,
                SUM(COALESCE(total_fouls ,0))  AS fouls,
                SUM(COALESCE(yellow_cards,0))  AS yellow_cards,
                SUM(COALESCE(red_cards   ,0))  AS red_cards,
                COUNT(*)                       AS matches_played
        FROM    match_info
        GROUP BY referee_name

        ON DUPLICATE KEY UPDATE
            fouls          = VALUES(fouls),
            yellow_cards   = VALUES(yellow_cards),
            red_cards      = VALUES(red_cards),
            matches_played = VALUES(matches_played);
        """
        DB.execute(sql)

    def train_context_ras_model(self):
        def flip(series: pd.Series) -> pd.Series:
            flipped = -series
            flipped[series == 0] = 0.0
            return flipped

        league_id_df = DB.select("SELECT league_id FROM league_data WHERE is_active = 1")

        for league_id in league_id_df['league_id'].tolist():
            sql_query = f"""
                SELECT 
                    mi.match_id,
                    mi.home_team_id,
                    mi.away_team_id,
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
                    md.minutes_played,
                    md.match_state,
                    md.match_segment,
                    md.player_dif,
                    (md.teamA_headers + md.teamA_footers) AS home_shots,
                    (md.teamB_headers + md.teamB_footers) AS away_shots
                FROM match_info mi
                JOIN match_detail md ON mi.match_id = md.match_id
                WHERE mi.league_id = %s
            """
            context_df = DB.select(sql_query, (league_id,))
            context_df['date'] = pd.to_datetime(context_df['date'])
            context_df['match_state'] = pd.to_numeric(context_df['match_state'], errors='raise').astype(float)
            context_df['player_dif']  = pd.to_numeric(context_df['player_dif'],  errors='raise').astype(float)

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
                'match_time'         : context_df['date'].apply(_bucket_time)
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
                'match_time'         : context_df['date'].apply(_bucket_time)
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
            
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=500,
                nfold=5,
                early_stopping_rounds=100,
                metrics='poisson-nloglik',
                verbose_eval=False
            )
            
            optimal_rounds = len(cv_results)
            booster = xgb.train(params, dtrain, num_boost_round=optimal_rounds)

            _save(f'ras_{league_id}', booster, X.columns.tolist())

    def train_refined_sq_model(self) -> tuple[xgb.Booster, list[str]]:
        sql = """
            SELECT
                total_plsqa,
                shooter_sq,
                assister_sq,
                CASE WHEN match_state < 0 THEN 'Trailing'
                     WHEN match_state = 0 THEN 'Level'
                     ELSE 'Leading' END AS match_state,
                CASE WHEN player_dif < 0 THEN 'Neg'
                     WHEN player_dif = 0 THEN 'Neu'
                     ELSE 'Pos' END      AS player_dif,
                xg
            FROM shots_data
            WHERE total_plsqa IS NOT NULL
        """
        df = DB.select(sql)

        cat_cols = ['match_state', 'player_dif']
        num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        for c in cat_cols:
            df[c] = df[c].astype(str)

        X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)
        X     = pd.concat([df[num_cols], X_cat], axis=1).astype(float)
        y     = df['xg'].astype(float)

        dtrain = xgb.DMatrix(X, label=y)
        params = dict(objective='reg:logistic',
                      eval_metric='logloss',
                      tree_method='hist',
                      max_depth=6,
                      eta=0.05,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      min_child_weight=2)

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=500,
            nfold=5,
            early_stopping_rounds=100,
            metrics='logloss',
            verbose_eval=False
        )
        
        optimal_rounds = len(cv_results)
        booster = xgb.train(params, dtrain, num_boost_round=optimal_rounds)

        _save('rsq', booster, X.columns.tolist())

    def train_post_shot_goal_model(self) -> tuple[xgb.Booster, list[str]]:
        sql = """
            SELECT
                sd.RSQ,
                sd.shooter_A,
                sd.GK_A,
                CASE WHEN sd.team_id = mi.home_team_id
                     THEN 1 ELSE 0 END                       AS team_is_home,
                CASE WHEN sd.team_id = mi.home_team_id
                     THEN mi.home_elevation_dif
                     ELSE mi.away_elevation_dif END          AS team_elevation_dif,
                CASE WHEN sd.team_id = mi.home_team_id
                     THEN 0 ELSE mi.away_travel END          AS team_travel,
                CASE WHEN sd.team_id = mi.home_team_id
                     THEN mi.home_rest_days
                     ELSE mi.away_rest_days END              AS team_rest_days,
                mi.temperature_c,
                mi.is_raining,
                mi.date,
                sd.outcome
            FROM   shots_data sd
            JOIN   match_info mi ON mi.match_id = sd.match_id
        """
        df = DB.select(sql)

        df['date']       = pd.to_datetime(df['date'])
        df['match_time'] = df['date'].apply(lambda t: 'aft' if 9 <= t.hour < 14
                                                      else ('evening' if 14 <= t.hour < 19
                                                            else 'night'))

        cat_cols  = ['match_time']
        bool_cols = ['team_is_home', 'is_raining']
        num_cols  = ['RSQ', 'shooter_A', 'GK_A',
                     'team_elevation_dif', 'team_travel',
                     'team_rest_days', 'temperature_c']

        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

        df[bool_cols] = (
            df[bool_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0) 
            .astype(int)           
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=num_cols)

        X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)
        X     = pd.concat([df[num_cols + bool_cols], X_cat], axis=1).astype(float)
        y     = df['outcome'].astype(int)

        pos_rate         = y.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        dtrain = xgb.DMatrix(X, label=y)
        params = dict(objective='binary:logistic',
                      eval_metric='logloss',
                      base_score = pos_rate,
                      scale_pos_weight = scale_pos_weight,
                      tree_method='hist',
                      max_depth=6,
                      eta=0.05,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      min_child_weight=2)
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1500,
            nfold=5,
            stratified=True,
            early_stopping_rounds=100,
            verbose_eval=False
        )

        optimal_rounds = len(cv_results)
        booster = xgb.train(params, dtrain, num_boost_round=optimal_rounds)

        _save('psxg', booster, X.columns.tolist())

# ------------------------------ Monte Carlo ------------------------------
class Alg:
    def __init__(self, schedule_id, home_initial_goals, away_initial_goals, match_initial_time, home_n_subs_avail, away_n_subs_avail):
        self.schedule_id = schedule_id

        match_df = DB.select("SELECT * FROM schedule_data WHERE schedule_id = %s", (self.schedule_id,))

        self.home_team_id = int(match_df.iloc[0]['home_team_id'])
        self.away_team_id = int(match_df.iloc[0]['away_team_id'])
        self.home_players_init_data = json.loads(match_df.iloc[0]['home_players_data'])
        self.away_players_init_data = json.loads(match_df.iloc[0]['away_players_data'])
        self.league_id = int(match_df.iloc[0]['league_id'])
        self.home_elevation_dif = int(match_df.iloc[0]['home_elevation_dif'])
        self.away_elevation_dif = int(match_df.iloc[0]['away_elevation_dif'])
        self.away_travel = int(match_df.iloc[0]['away_travel'])
        self.home_rest_days = int(match_df.iloc[0]['home_rest_days'])
        self.away_rest_days = int(match_df.iloc[0]['away_rest_days'])
        self.temperature = int(match_df.iloc[0]['temperature'])
        self.is_raining = int(match_df.iloc[0]['is_raining'])
        self.home_initial_goals = home_initial_goals
        self.away_initial_goals = away_initial_goals
        self.match_initial_time = match_initial_time
        self.match_time_label = _bucket_time(self.match_initial_time)
        self.home_n_subs_avail = home_n_subs_avail
        self.away_n_subs_avail = away_n_subs_avail
        self.referee_name = match_df.iloc[0]['referee_name']

        baseline_df = DB.select("SELECT sh_baseline_coef, hxg_baseline_coef, fxg_baseline_coef FROM league_data WHERE league_id = %s", (self.league_id,)
        )
        self.sh_baseline_coef = float(baseline_df.iloc[0]['sh_baseline_coef']) if not baseline_df.empty and baseline_df.iloc[0]['sh_baseline_coef'] is not None else 0.0
        self.hxg_baseline_coef = float(baseline_df.iloc[0]['hxg_baseline_coef']) if not baseline_df.empty and baseline_df.iloc[0]['hxg_baseline_coef'] is not None else 0.0
        self.fxg_baseline_coef = float(baseline_df.iloc[0]['fxg_baseline_coef']) if not baseline_df.empty and baseline_df.iloc[0]['fxg_baseline_coef'] is not None else 0.0

        self.ras_booster, self.ras_cr_columns = self._load_model(f'ras_{self.league_id}')
        self.ctx_mult_home, self.ctx_mult_away = self.precompute_ctx_multipliers()
        self.rsq_booster, self.rsq_columns     = self._load_model('rsq')
        self.rsq_pred_cache = {}
        self.rsq_col_idx    = {c: i for i, c in enumerate(self.rsq_columns)}
        self.psxg_booster, self.psxg_columns   = self._load_model('psxg')
        self.psxg_pred_cache = {}
        self.psg_col_idx     = {c: i for i, c in enumerate(self.psxg_columns)}
        self.ref_stats = self.get_referee_stats()
        self.precompute_card_sim_data()

        self.home_starters, self.home_subs = self.divide_matched_players(self.home_players_init_data)
        self.away_starters, self.away_subs = self.divide_matched_players(self.away_players_init_data)

        self.home_players_data = self.get_players_data(self.home_team_id, self.home_starters, self.home_subs, self.home_players_init_data)
        self.away_players_data = self.get_players_data(self.away_team_id, self.away_starters, self.away_subs, self.away_players_init_data)

        self._base_home_players_data = copy.deepcopy(self.home_players_data)
        self._base_away_players_data = copy.deepcopy(self.away_players_data)

        self.home_sub_minutes, self.away_sub_minutes = self.get_sub_minutes(self.home_team_id, self.away_team_id, self.match_initial_time, self.home_n_subs_avail, self.away_n_subs_avail)
        self.all_sub_minutes = list(set(list(self.home_sub_minutes.keys()) + list(self.away_sub_minutes.keys())))

        if self.match_initial_time >= 45:
            range_value = 2000
        elif self.match_initial_time < 1:
            range_value = 8000
        elif self.match_initial_time < 45:
            range_value = 6000

        self.run_simulations(range_value, 4)

    def _simulate_single(self, i):
        self.home_players_data = copy.deepcopy(self._base_home_players_data)
        self.away_players_data = copy.deepcopy(self._base_away_players_data)

        home_goals = self.home_initial_goals
        away_goals = self.away_initial_goals
        home_active_players  = self.home_starters.copy()
        away_active_players  = self.away_starters.copy()
        home_passive_players = self.home_subs.copy()
        away_passive_players = self.away_subs.copy()

        shot_rows = []
        card_rows = []

        home_status, away_status = self.get_status(home_goals, away_goals)
        time_segment = self.get_time_segment(self.match_initial_time)

        home_ras, home_rahs, home_rafs, home_plhsq, home_plfsq = self.get_teams_ra(home_active_players, away_active_players, self.home_players_data, self.away_players_data)
        away_ras, away_rahs, away_rafs, away_plhsq, away_plfsq = self.get_teams_ra(away_active_players, home_active_players, self.away_players_data, self.home_players_data)
        home_players_prob = self.build_player_probs(home_active_players, self.home_players_data)
        away_players_prob = self.build_player_probs(away_active_players, self.away_players_data)

        home_mult = self.ctx_mult_home[(home_status, time_segment, 0)]
        away_mult = self.ctx_mult_away[(away_status, time_segment, 0)]
        home_context_ras = max(1e-6, home_ras) * home_mult
        away_context_ras = max(1e-6, away_ras) * away_mult

        home_psxg_cache = self.build_psxg_cache(home_active_players, self.home_players_data,
                                                home_plhsq, home_plfsq,
                                                home_status,  0,
                                                True, self.away_players_data)
        away_psxg_cache = self.build_psxg_cache(away_active_players, self.away_players_data,
                                                away_plhsq, away_plfsq,
                                                away_status, 0,
                                                False, self.home_players_data)
        
        home_foul_p = self.get_team_foul_prob(home_active_players,
                                                away_active_players,
                                                home_status,
                                                is_home=True)

        away_foul_p = self.get_team_foul_prob(away_active_players,
                                                home_active_players,
                                                away_status,
                                                is_home=False)

        context_ras_change = False
        for minute in range(self.match_initial_time, 91):
            home_status, away_status = self.get_status(home_goals, away_goals)
            time_segment = self.get_time_segment(minute)
            if minute in [16, 31, 46, 61, 76]:
                context_ras_change = True

            if minute in self.all_sub_minutes:
                context_ras_change = True
                if minute in list(self.home_sub_minutes.keys()):
                    home_active_players, home_passive_players = self.swap_players(home_active_players, home_passive_players, self.home_players_data, self.home_sub_minutes[minute], home_status)
                if minute in list(self.away_sub_minutes.keys()):
                    away_active_players, away_passive_players = self.swap_players(away_active_players, away_passive_players, self.away_players_data, self.away_sub_minutes[minute], away_status)
                home_ras, home_rahs, home_rafs, home_plhsq, home_plfsq = self.get_teams_ra(home_active_players, away_active_players, self.home_players_data, self.away_players_data)
                away_ras, away_rahs, away_rafs, away_plhsq, away_plfsq = self.get_teams_ra(away_active_players, home_active_players, self.away_players_data, self.home_players_data)
                home_players_prob = self.build_player_probs(home_active_players, self.home_players_data)
                away_players_prob = self.build_player_probs(away_active_players, self.away_players_data)

            if context_ras_change:
                context_ras_change = False
                home_mult = self.ctx_mult_home[(home_status, time_segment, 0)]
                away_mult = self.ctx_mult_away[(away_status, time_segment, 0)]
                home_context_ras = max(1e-6, home_ras) * home_mult
                away_context_ras = max(1e-6, away_ras) * away_mult

                home_psxg_cache = self.build_psxg_cache(home_active_players, self.home_players_data,
                                                    home_plhsq, home_plfsq,
                                                    home_status,  0,
                                                    True, self.away_players_data)
                away_psxg_cache = self.build_psxg_cache(away_active_players, self.away_players_data,
                                                    away_plhsq, away_plfsq,
                                                    away_status, 0,
                                                    False, self.home_players_data)
                
                home_foul_p = self.get_team_foul_prob(home_active_players,
                                                        away_active_players,
                                                        home_status,
                                                        is_home=True)

                away_foul_p = self.get_team_foul_prob(away_active_players,
                                                        home_active_players,
                                                        away_status,
                                                        is_home=False)

            home_shots = np.random.poisson(home_context_ras)
            away_shots = np.random.poisson(away_context_ras)

            if home_shots:
                for _ in range(home_shots):
                    body_part = self.get_shot_type(home_rahs, home_rafs)
                    shooter = self.get_shooter(home_players_prob, body_part)
                    assister = self.get_assister(home_players_prob, body_part, shooter)
                    xg_prob   = home_psxg_cache.get((shooter, assister, body_part), 0.0)
                    outcome = int(np.random.rand() < xg_prob)
                    # print(f"({shooter}, {assister}, {body_part}) = {xg_prob} = {outcome}")
                    if outcome == 1:
                        home_goals += 1
                        context_ras_change = True
                    shot_rows.append((i, minute, shooter, self.home_team_id, outcome, body_part, assister))

            if away_shots:
                for _ in range(away_shots):
                    body_part = self.get_shot_type(away_rahs, away_rafs)
                    shooter = self.get_shooter(away_players_prob, body_part)
                    assister = self.get_assister(away_players_prob, body_part, shooter)
                    xg_prob   = away_psxg_cache.get((shooter, assister, body_part), 0.0)
                    outcome = int(np.random.rand() < xg_prob) 
                    # print(f"({shooter}, {assister}, {body_part}) = {xg_prob} = {outcome}")
                    if outcome == 1:
                        away_goals += 1
                        context_ras_change = True
                    shot_rows.append((i, minute, shooter, self.away_team_id, outcome, body_part, assister)) 

            home_fouls = np.random.poisson(home_foul_p)
            for _ in range(home_fouls):
                fouler     = self.choose_fouler(home_active_players, self.home_players_data)
                card_type  = self.determine_card(fouler, self.home_players_data)
                if card_type != 'NONE':
                    card_rows.append((i, minute, fouler, self.home_team_id, card_type))
                if card_type == 'YC':
                    self.home_players_data[fouler]['sim_yellow'] += 1
                    if self.home_players_data[fouler]['sim_yellow'] >= 2:
                        home_active_players.remove(fouler)
                        context_ras_change = True
                elif card_type == 'RC':
                    self.home_players_data[fouler]['sim_red'] = True
                    if fouler in home_active_players:
                        home_active_players.remove(fouler)
                        context_ras_change = True

            away_fouls = np.random.poisson(away_foul_p)
            for _ in range(away_fouls):
                fouler     = self.choose_fouler(away_active_players, self.away_players_data)
                card_type  = self.determine_card(fouler, self.away_players_data)
                if card_type != 'NONE':
                    card_rows.append((i, minute, fouler, self.away_team_id, card_type))
                if card_type == 'YC':
                    self.away_players_data[fouler]['sim_yellow'] += 1
                    if self.away_players_data[fouler]['sim_yellow'] >= 2:
                        away_active_players.remove(fouler)
                        context_ras_change = True
                elif card_type == 'RC':
                    self.away_players_data[fouler]['sim_red'] = True
                    if fouler in away_active_players:
                        away_active_players.remove(fouler)
                        context_ras_change = True
        return shot_rows, card_rows

    def run_simulations(self, n_sims, n_workers, flush_every: int = 1000):
        if n_workers is None:
            n_workers = os.cpu_count() or 1

        shot_buf, card_buf = [], []
        first_flush_done = False

        def _flush():
            nonlocal first_flush_done
            if not shot_buf:
                return
            self.insert_sim_data(
                shot_buf,
                self.schedule_id,
                initial_delete=not first_flush_done
            )
            first_flush_done = True
            shot_buf.clear()
            card_buf.clear()

        if n_workers > 1:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for idx, (s, c) in enumerate(
                        tqdm(pool.imap_unordered(self._simulate_single, range(n_sims)),
                            total=n_sims,
                            desc=f'Simulations ({n_workers} workers)')):
                    shot_buf.extend(s)
                    card_buf.extend(c)
                    if (idx + 1) % flush_every == 0:
                        _flush()
        else:
            for idx in tqdm(range(n_sims), desc='Simulations (1 worker)'):
                s, c = self._simulate_single(idx)
                shot_buf.extend(s)
                card_buf.extend(c)
                if (idx + 1) % flush_every == 0:
                    _flush()

        _flush()

    def predict_context_ras(self, booster, feature_columns, new_match, *, raw=False):
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
        prediction = booster.predict(dmatrix, output_margin=raw)
        return prediction[0]

    def precompute_ctx_multipliers(self):
        def _template(is_home: bool):
            return {
                'team_is_home'     : int(is_home),
                'team_elevation_dif': self.home_elevation_dif if is_home else self.away_elevation_dif,
                'opp_elevation_dif' : self.away_elevation_dif if is_home else self.home_elevation_dif,
                'team_travel'       : 0 if is_home else self.away_travel,
                'opp_travel'        : self.away_travel if is_home else 0,
                'team_rest_days'    : self.home_rest_days if is_home else self.away_rest_days,
                'opp_rest_days'     : self.away_rest_days if is_home else self.home_rest_days,
                'temperature_c'     : self.temperature,
                'is_raining'        : self.is_raining,
                'match_time'        : self.match_time_label,
                'total_ras'         : 1.0          # 1 ⇒  log(1)=0  ⇒  pure model effect
            }

        states       = [-1.5, -1, 0, 1, 1.5]
        segments     = [1, 2, 3, 4, 5, 6]
        player_diffs = [-1.5, -1, 0, 1, 1.5]

        home_cache, away_cache = {}, {}
        for st, sg, pdif in itertools.product(states, segments, player_diffs):
            for is_home, cache in ((True, home_cache), (False, away_cache)):
                row = _template(is_home)
                row.update({'match_state': st,
                            'match_segment': sg,
                            'player_dif'  : pdif})
                # raw=True ⇒ get the margin only
                raw_margin = self.predict_context_ras(self.ras_booster,
                                                      self.ras_cr_columns,
                                                      pd.DataFrame([row]),
                                                      raw=True)
                cache[(st, sg, pdif)] = np.exp(raw_margin)
        return home_cache, away_cache

    def _predict_refined_sq_bulk(self, df: pd.DataFrame) -> np.ndarray:
        cat_cols = ['match_state', 'player_dif']
        num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']

        missing = [c for c in num_cols + cat_cols if c not in df.columns]
        if missing:
            raise KeyError(f'Refined-SQ prediction failed; missing columns: {missing}')

        n_rows = len(df)
        X = np.zeros((n_rows, len(self.rsq_columns)), dtype=np.float32)

        for col in num_cols:
            X[:, self.rsq_col_idx[col]] = (
                pd.to_numeric(df[col], errors='coerce')
                  .fillna(0)
                  .to_numpy(dtype=np.float32)
            )

        for col in cat_cols:
            pref = f'{col}_'
            for i, v in enumerate(df[col].astype(str).fillna('nan')):
                idx = self.rsq_col_idx.get(f'{pref}{v}')
                if idx is not None:
                    X[i, idx] = 1.0

        return self.rsq_booster.inplace_predict(X)

    def _predict_post_shot_bulk(self, df: pd.DataFrame) -> np.ndarray:
        cat_cols  = ['match_time']
        bool_cols = ['team_is_home', 'is_raining']
        num_cols  = ['RSQ', 'shooter_A', 'GK_A',
                    'team_elevation_dif', 'team_travel',
                    'team_rest_days', 'temperature_c']

        X_num_bool = df[num_cols + bool_cols].astype(float)
        X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)
        X_all = pd.concat([X_num_bool, X_cat], axis=1)
        X_all = X_all.reindex(columns=self.psxg_columns, fill_value=0).astype(float)
        dmatrix = xgb.DMatrix(X_all)
        return self.psxg_booster.predict(dmatrix)

    def build_xg_cache(self,
                       active_ids      : list[int],
                       players_df      ,
                       plsqa_head      : float,
                       plsqa_foot      : float,
                       match_state_num : int,
                       player_dif_num  : int) -> dict:

        def _safe_sq(src, pid, fallback: float) -> float:
            if isinstance(src, pd.DataFrame):
                if pid in src.index:
                    if 'sq' in src.columns:
                        return float(src.at[pid, 'sq'])
                    if 'shooter_sq' in src.columns:
                        return float(src.at[pid, 'shooter_sq'])
            elif isinstance(src, dict):
                rec = src.get(pid, {})
                if isinstance(rec, dict) and 'sq' in rec:
                    return float(rec['sq'])
            return fallback

        state = 'Trailing' if match_state_num < 0 else 'Leading' if match_state_num > 0 else 'Level'
        pdif  = 'Neg'      if player_dif_num  < 0 else 'Pos'     if player_dif_num  > 0 else 'Neu'

        assist_pool = [None] + active_ids
        cache_keys, new_rows, out = [], [], {}

        for shooter in active_ids:
            for body, plsqa in (('Head', plsqa_head), ('Foot', plsqa_foot)):
                base_sq_default = (self.hxg_baseline_coef if body == 'Head'
                                   else self.fxg_baseline_coef)

                shooter_sq = _safe_sq(players_df, shooter, base_sq_default)

                for assister in assist_pool:
                    assister_sq = 0.0 if assister is None else \
                                  _safe_sq(players_df, assister, base_sq_default)

                    key = (round(plsqa, 4), shooter_sq, assister_sq, state, pdif)
                    cache_keys.append((shooter, assister, body, key))

                    if key not in self.rsq_pred_cache:
                        new_rows.append(dict(total_plsqa=plsqa,
                                             shooter_sq=shooter_sq,
                                             assister_sq=assister_sq,
                                             match_state=state,
                                             player_dif=pdif))

        if new_rows:
            preds = self._predict_refined_sq_bulk(pd.DataFrame(new_rows))
            for k, p in zip([ck[-1] for ck in cache_keys if ck[-1] not in self.rsq_pred_cache],
                            preds):
                self.rsq_pred_cache[k] = float(p)

        for shooter, assister, body, k in cache_keys:
            out[(shooter, assister, body)] = self.rsq_pred_cache[k]

        return out

    def build_psxg_cache(self,
                         active_ids      : list[int],
                         players_df      ,
                         plsqa_head      : float,
                         plsqa_foot      : float,
                         match_state_num : int,
                         player_dif_num  : int,
                         is_home         : bool,
                         opp_players_df  ) -> dict:

        rsq_cache = self.build_xg_cache(active_ids, players_df,
                                        plsqa_head, plsqa_foot,
                                        match_state_num, player_dif_num)

        def _safe(src, pid, col):
            if isinstance(src, pd.DataFrame):
                if pid in src.index and col in src.columns:
                    return float(src.at[pid, col])
                if 'player_id' in src.columns:
                    row = src[src['player_id'] == pid]
                    return float(row.iloc[0][col]) if not row.empty else 0.0
                return 0.0
            rec = src.get(pid, {})
            return float(rec.get(col, 0.0)) if isinstance(rec, dict) else 0.0

        if isinstance(opp_players_df, pd.DataFrame):
            gk_rows = opp_players_df.loc[opp_players_df.get('position') == 'GK']
            gk_ability = float(gk_rows['GK_A'].iloc[0]) if not gk_rows.empty else 0.0
        else:
            gk_id = next((pid for pid, rec in opp_players_df.items()
                          if rec.get('position') == 'GK'), None)
            gk_ability = float(opp_players_df.get(gk_id, {}).get('GK_A', 0.0)) if gk_id else 0.0

        team_elev   = self.home_elevation_dif if is_home else self.away_elevation_dif
        team_travel = 0.0 if is_home else self.away_travel
        team_rest   = self.home_rest_days  if is_home else self.away_rest_days
        match_time  = self.match_time_label

        cache_keys, new_rows, out = [], [], {}
        assist_pool = [None] + active_ids

        for shooter in active_ids:
            shooter_ability = _safe(players_df, shooter, 'shooter_A')
            for assister in assist_pool:
                for body, rsq in (('Head', rsq_cache.get((shooter, assister, 'Head'), 0.0)),
                                  ('Foot', rsq_cache.get((shooter, assister, 'Foot'), 0.0))):
                    key = (round(rsq, 6), shooter_ability, gk_ability, is_home, body)
                    cache_keys.append((shooter, assister, body, key))
                    if key not in self.psxg_pred_cache:
                        new_rows.append(dict(
                            RSQ=rsq,
                            shooter_A=shooter_ability,
                            GK_A=gk_ability,
                            team_is_home=int(is_home),
                            team_elevation_dif=team_elev,
                            team_travel=team_travel,
                            team_rest_days=team_rest,
                            temperature_c=self.temperature,
                            is_raining=int(self.is_raining),
                            match_time=match_time
                        ))

        if new_rows:
            preds = self._predict_post_shot_bulk(pd.DataFrame(new_rows))
            for k, p in zip([ck[-1] for ck in cache_keys if ck[-1] not in self.psxg_pred_cache], preds):
                self.psxg_pred_cache[k] = float(p)

        for shooter, assister, body, k in cache_keys:
            out[(shooter, assister, body)] = self.psxg_pred_cache[k]

        return out
    
    def divide_matched_players(self, players_data):
        starters = [p['player_id'] for p in players_data if p['on_field']]
        subs = [p['player_id'] for p in players_data if p['bench']]
        return starters, subs

    def get_players_data(self, team_id, team_starters, team_subs, initial_data=None):
        all_players = team_starters + team_subs

        escaped_players = [player.replace("'", "''") for player in all_players]

        team_player_str = ", ".join([f"'{player}'" for player in escaped_players])

        sql_query = f"""
            SELECT 
                *
            FROM players_data
            WHERE current_team = '{team_id}'
            AND player_id IN ({team_player_str});
        """
        players_df = DB.select(sql_query)
        numeric_cols = ['sub_in', 'sub_out']
        for col in numeric_cols:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)

        initial_mapping = {}
        if initial_data is not None:
            initial_mapping = {player['player_id']: player for player in initial_data}

        players_dict = {}

        for player_id in players_df['player_id'].unique():
            player_rows = players_df[players_df['player_id'] == player_id]
            player_info = player_rows.iloc[0].to_dict()

            def _status_prob(raw_status):
                if isinstance(raw_status, str):
                    try:
                        counts = ast.literal_eval(raw_status)
                    except (ValueError, SyntaxError):
                        counts = {}
                elif isinstance(raw_status, dict):
                    counts = raw_status
                else:
                    counts = {}

                counts = {k.title(): v for k, v in counts.items()}
                base = {'Leading': 0, 'Level': 0, 'Trailing': 0}
                base.update(counts)

                total = sum(base.values())
                return {k: v / total if total else 0 for k, v in base.items()}

            player_info['in_status_prob'] = _status_prob(player_info.get('in_status'))
            player_info['out_status_prob'] = _status_prob(player_info.get('out_status'))

            players_dict[player_id] = player_info
            
            if player_id in initial_mapping:
                extracted = initial_mapping[player_id]
                players_dict[player_id]['sim_yellow'] = 1 if extracted.get('yellow_card') else 0
                players_dict[player_id]['sim_red'] = extracted.get('red_card', False)
            else:
                players_dict[player_id]['sim_yellow'] = 1 if player_info.get('yellow_card') else 0
                players_dict[player_id]['sim_red'] = player_info.get('red_card', False)

        return players_dict

    def get_sub_minutes(self, home_id, away_id, match_initial_time, home_n_subs_avail, away_n_subs_avail):
        teams_data_query = f"""
            SELECT 
                mb.match_id,
                mb.sub_in,
                pd.current_team AS team_id
            FROM match_breakdown mb
            JOIN match_info mi ON mb.match_id = mi.match_id
            JOIN players_data pd ON mb.player_id = pd.player_id
            WHERE (mi.home_team_id IN ({home_id}, {away_id}) OR mi.away_team_id IN ({home_id}, {away_id}));
        """

        query_df = DB.select(teams_data_query)
        valid_subs_df = query_df[(query_df['sub_in'].notnull()) & (query_df['sub_in'] != 0)]

        home_avg_subs = round(valid_subs_df[valid_subs_df['team_id'] == home_id].groupby('match_id').size().mean())
        away_avg_subs = round(valid_subs_df[valid_subs_df['team_id'] == away_id].groupby('match_id').size().mean())

        effective_home_subs = max(0, min(home_avg_subs - (5 - home_n_subs_avail), home_n_subs_avail))
        effective_away_subs = max(0, min(away_avg_subs - (5 - away_n_subs_avail), away_n_subs_avail))

        def get_distribution(team_id, avail_subs):
            if avail_subs == 0:
                return {100: 0}
            if avail_subs == 1:
                n_windows = 1
            elif avail_subs < 5:
                n_windows = 2
            else:
                n_windows = 3

            top_minutes = (valid_subs_df[(valid_subs_df['team_id'] == team_id) & (valid_subs_df['sub_in'] > match_initial_time)]['sub_in'].value_counts().head(n_windows).index.tolist())

            base = avail_subs // n_windows
            remainder = avail_subs % n_windows
            distribution = {}
            for i in range(n_windows):
                distribution[top_minutes[i]] = base + 1 if i < remainder else base
            return distribution

        home_distribution = get_distribution(home_id, effective_home_subs)
        away_distribution = get_distribution(away_id, effective_away_subs)

        return home_distribution, away_distribution

    def swap_players(self, active_players, passive_players, players_data, subs, game_status_n):
        def interpret_game_status(status_code):
            if status_code > 0:
                return "Leading"
            elif status_code < 0:
                return "Trailing"
            else:
                return "Level"
            
        game_status = interpret_game_status(game_status_n)

        total_active_minutes = 0
        for player in active_players:
            total_active_minutes += players_data[player]['minutes_played']

        active_players_dict = {}

        for player in active_players:
            active_players_dict[player] = (1 - (players_data[player]['minutes_played'] / total_active_minutes)) * (players_data[player]['out_status_prob'][game_status])

        total_active_p = sum(active_players_dict.values())
        if total_active_p == 0:
            num_players = len(active_players_dict)
            normalized_active_p = {key: 1.0 / num_players for key in active_players_dict.keys()}
        else:
            normalized_active_p = {key: value / total_active_p for key, value in active_players_dict.items()}
            probabilities = list(normalized_active_p.values())
            if subs > 1 and probabilities.count(1.0) == 1:
                max_index = probabilities.index(1.0)
                probabilities[max_index] = 0.99
                small_probability = 0.01 / (len(probabilities) - 1)
                for i in range(len(probabilities)):
                    if i != max_index:
                        probabilities[i] = small_probability
            for i, key in enumerate(normalized_active_p.keys()):
                normalized_active_p[key] = probabilities[i]

        active_weights = list(normalized_active_p.values())     

        picked_out_players = np.random.choice(active_players, p=active_weights, replace=False, size=subs)

        total_passive_minutes = 0
        for player in passive_players:
            total_passive_minutes += players_data[player]['minutes_played']

        passive_players_dict = {}

        for player in passive_players:
            passive_players_dict[player] = (players_data[player]['minutes_played'] / total_passive_minutes) * (players_data[player]['in_status_prob'][game_status])

        total_passive_p = sum(passive_players_dict.values())
        if total_passive_p == 0:
            num_players = len(passive_players_dict)
            normalized_passive_p = {key: 1.0 / num_players for key in passive_players_dict.keys()}
        else:
            normalized_passive_p = {key: value / total_passive_p for key, value in passive_players_dict.items()}
            probabilities = list(normalized_passive_p.values())
            if subs > 1 and probabilities.count(1.0) == 1:
                max_index = probabilities.index(1.0)
                probabilities[max_index] = 0.99
                small_probability = 0.01 / (len(probabilities) - 1)
                for i in range(len(probabilities)):
                    if i != max_index:
                        probabilities[i] = small_probability
            for i, key in enumerate(normalized_passive_p.keys()):
                normalized_passive_p[key] = probabilities[i]

        passive_weights = list(normalized_passive_p.values())   

        picked_in_players = np.random.choice(passive_players, p=passive_weights, replace=False, size=subs)

        active_players = [player for player in active_players if player not in picked_out_players]
        active_players.extend(picked_in_players)
        passive_players = [player for player in passive_players if player not in picked_in_players]

        return active_players, passive_players

    def get_teams_ra(self, offensive_players, defensive_players, offensive_data, defensive_data):
        # team_total_ras
        team_off_ras = sum(offensive_data[p]['off_sh_coef'] for p in offensive_players)
        opp_def_ras  = sum(defensive_data[p]['def_sh_coef'] for p in defensive_players)
        team_total_ras = self.sh_baseline_coef + team_off_ras - opp_def_ras

        # team_rahs
        team_off_rahs = sum(offensive_data[p]['off_headers_coef'] for p in offensive_players)
        opp_def_rahs  = sum(defensive_data[p]['def_headers_coef'] for p in defensive_players)
        team_rahs = team_off_rahs - opp_def_rahs

        # team_rafs
        team_off_rafs = sum(offensive_data[p]['off_footers_coef'] for p in offensive_players)
        opp_def_rafs  = sum(defensive_data[p]['def_footers_coef'] for p in defensive_players)
        team_rafs = team_off_rafs - opp_def_rafs

        # team_plhsq
        team_off_plhsq = sum(offensive_data[p]['off_hxg_coef'] for p in offensive_players)
        opp_def_plhsq  = sum(defensive_data[p]['def_hxg_coef'] for p in defensive_players)
        team_plhsq = self.hxg_baseline_coef + team_off_plhsq - opp_def_plhsq

        # team_plfsq
        team_off_plfsq = sum(offensive_data[p]['off_fxg_coef'] for p in offensive_players)
        opp_def_plfsq  = sum(defensive_data[p]['def_fxg_coef'] for p in defensive_players)
        team_plfsq = self.fxg_baseline_coef  +team_off_plfsq - opp_def_plfsq

        return team_total_ras, team_rahs, team_rafs, team_plhsq, team_plfsq
 
    def get_status(self, home_goals, away_goals):
        diff = home_goals - away_goals
        if diff == 0:
            return 0.0, 0.0
        elif diff == 1:
            return 1.0, -1.0
        elif diff > 1:
            return 1.5, -1.5
        elif diff == -1:
            return -1.0, 1.0
        elif diff < -1:
            return -1.5, 1.5

    def get_time_segment(self, minute):
        if minute < 15:
            return 1
        elif minute < 30:
            return 2
        elif minute < 45:
            return 3
        elif minute < 60:
            return 4
        elif minute < 75:
            return 5
        else:
            return 6
        
    def get_shot_type(self, rahs, rafs):
        sharpness = 50
        delta = rafs - rahs
        foot_prob = 1 / (1 + np.exp(-sharpness * delta))
        head_prob = 1 - foot_prob

        probs = [head_prob, foot_prob]

        selected_index = np.random.choice([0, 1], p=probs)

        if selected_index == 0:
            body_part = "Head"
        else:
            body_part = "Foot"
        return body_part
    
    def build_player_probs(self, active_players, players_data):
        def _normalise(rate_dict):
            tot = sum(rate_dict.values())
            if tot == 0:
                n = len(rate_dict)
                return {k: 1 / n for k in rate_dict}
            return {k: v / tot for k, v in rate_dict.items()}

        rates = {
            'headers': {},
            'footers': {},
            'non_assisted_footers': {},
            'key_passes': {}
        }

        for player in active_players:
            data    = players_data[player]
            minutes = max(1, data.get('minutes_played', 1))

            rates['headers'][player]              = data.get('headers', 0) / minutes
            rates['footers'][player]              = data.get('footers', 0) / minutes
            rates['non_assisted_footers'][player] = data.get('non_assisted_footers', 0) / minutes
            rates['key_passes'][player]              = data.get('key_passes', 0) / minutes

        shooter_prob = {
            'headers': _normalise(rates['headers']),
            'footers': _normalise(rates['footers'])
        }

        assist_prob = {'headers': {}, 'footers': {}}

        for shooter in active_players:
            dist_f = {}
            dist_f[None] = rates['non_assisted_footers'][shooter]
            for assister in active_players:
                if assister == shooter:
                    continue
                dist_f[assister] = rates['key_passes'][assister]
            assist_prob['footers'][shooter] = _normalise(dist_f)

            dist_h = {}
            for assister in active_players:
                if assister == shooter:
                    continue
                dist_h[assister] = rates['key_passes'][assister]
            assist_prob['headers'][shooter] = _normalise(dist_h)

        return {'shooter': shooter_prob, 'assist': assist_prob}

    def get_shooter(self, prob_dicts, body_part):
        _body_part_key = {'Head': 'headers', 'Foot': 'footers'}

        key      = _body_part_key[body_part] 
        probs    = prob_dicts['shooter'][key]
        players  = list(probs.keys())
        p_vals   = list(probs.values())
        return np.random.choice(players, p=p_vals)
    
    def get_assister(self, prob_dicts, body_part, shooter):
        _body_part_key = {'Head': 'headers', 'Foot': 'footers'}
        key      = _body_part_key[body_part]
        probs    = prob_dicts['assist'][key][shooter]
        ass      = list(probs.keys())
        p_vals   = list(probs.values())
        return np.random.choice(ass, p=p_vals)

    def insert_sim_data(self, rows, schedule_id, *, initial_delete: bool = False):
        if initial_delete:
            DB.execute("DELETE FROM simulation_data WHERE schedule_id = %s",
                    (schedule_id,))

        batch_size = 200
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            placeholders = ', '.join(['(%s, %s, %s, %s, %s, %s, %s, %s)'] * len(chunk))
            insert_sql = f"""
            INSERT INTO simulation_data 
                (sim_id, schedule_id, minute, shooter, squad, outcome, body_part, assister)
            VALUES {placeholders}
            """
            params = []
            for row in chunk:
                params.extend([row[0], schedule_id] + list(row[1:]))

            DB.execute(insert_sql, params)

    def get_referee_stats(self):
        sql = f"""
            SELECT fouls, yellow_cards, red_cards, matches_played
            FROM referee_data
            WHERE referee_name = '{self.referee_name}'
        """
        df = DB.select(sql)
        if df.empty:
            return {'fouls': 26.5, 'yellow_cards': 3.8, 'red_cards': 0.14, 'matches_played': 1}
        return df.iloc[0].to_dict()

    def precompute_card_sim_data(self):
        self.team_factor   = {True: 0.95, False: 1.05}
        self.status_factor = {'Leading': 0.88, 'Level': 1.0, 'Trailing': 1.11,
                              1: 0.88, 0: 1.0, -1: 1.11}      # handles numeric status too

        rf                 = max(1, self.ref_stats['matches_played'])
        self.ref_fouls_pm  = self.ref_stats['fouls']        / rf
        self.ref_ycs_pm    = self.ref_stats['yellow_cards'] / rf
        self.ref_rcs_pm    = self.ref_stats['red_cards']    / rf

        self.yc_prob_given_foul   = self.ref_ycs_pm / max(1e-5, self.ref_fouls_pm)
        self.rc_prob_given_foul   = self.ref_rcs_pm / max(1e-5, self.ref_fouls_pm)
        self.none_prob_given_foul = max(0.0, 1.0 - self.yc_prob_given_foul - self.rc_prob_given_foul)

        self.foul_prob_cache = {}

    def _calc_team_fouls_per90(self, active_players, opponent_players, players_data, opp_data):
        minutes_team = sum(players_data[p]['minutes_played'] for p in active_players) or 1
        minutes_opp  = sum(opp_data[p]['minutes_played']    for p in opponent_players) or 1

        commits_per90 = sum(
            (players_data[p]['fouls_committed'] / max(1, players_data[p]['minutes_played'])) * 90
            for p in active_players
        )
        drawn_per90 = sum(
            (opp_data[p]['fouls_drawn'] / max(1, opp_data[p]['minutes_played'])) * 90
            for p in opponent_players
        )

        return (commits_per90 + drawn_per90) / 2.0

    def get_team_foul_prob(self, active_players, opponent_players, status, is_home):
        if isinstance(status, (int, float)):
            status = 1 if status > 0 else -1 if status < 0 else 0
        key = (frozenset(active_players), frozenset(opponent_players), status, is_home)
        if key not in self.foul_prob_cache:
            players_data = self.home_players_data if is_home else self.away_players_data
            opp_data     = self.away_players_data if is_home else self.home_players_data

            team_f90  = self._calc_team_fouls_per90(active_players, opponent_players,
                                                    players_data, opp_data)
            
            opp_f90  = self._calc_team_fouls_per90(opponent_players,
                                                active_players,
                                                opp_data, players_data)

            sum_f90      = team_f90 + opp_f90
            normaliser   = (sum_f90 + self.ref_fouls_pm) / 2.0
            adjust_fac   = team_f90 / max(1e-5, normaliser)

            raw_per_min = team_f90 / 90.0 

            per_min = raw_per_min * adjust_fac * self.team_factor[is_home] * self.status_factor[status]

            self.foul_prob_cache[key] = max(per_min, 1e-6)   # keep ≥ very small
        return self.foul_prob_cache[key]
    
    def choose_fouler(self, active_players, players_dict):
        weights = [(players_dict[p]['fouls_committed'] / max(1, players_dict[p]['minutes_played']))
                   for p in active_players]
        total = sum(weights)
        if total == 0:
            weights = [1 / len(active_players)] * len(active_players)
        else:
            weights = [w / total for w in weights]
        return np.random.choice(active_players, p=weights)
    
    def determine_card(self, player_id, players_dict, k: int = 10):
        pdata = players_dict[player_id]

        fouls = pdata.get('fouls_committed', 0)
        ycs   = pdata.get('yellow_cards',     0)
        rcs   = pdata.get('red_cards',        0)

        player_yc_rate = (ycs + k * self.yc_prob_given_foul) / (fouls + k)
        player_rc_rate = (rcs + k * self.rc_prob_given_foul) / (fouls + k)

        weight_player = 0.5
        weight_ref    = 1.0 - weight_player

        yc_prob = weight_player * player_yc_rate + weight_ref * self.yc_prob_given_foul
        rc_prob = weight_player * player_rc_rate + weight_ref * self.rc_prob_given_foul

        total = yc_prob + rc_prob
        if total > 1.0:
            yc_prob /= total
            rc_prob /= total
            total = 1.0

        none_prob = 1.0 - total 
        probs     = [yc_prob, rc_prob, none_prob]

        probs = [max(p, 0.0) for p in probs]
        probs = np.array(probs) / np.sum(probs)

        outcome = np.random.choice(['YC', 'RC', 'NONE'], p=probs)
        return outcome

    @staticmethod
    def _load_model(name: str) -> tuple[xgb.Booster, list[str]]:
        booster_path  = f'models/{name}_booster.json'
        columns_path  = f'models/{name}_columns.json'

        booster = xgb.Booster()
        booster.load_model(booster_path)

        with open(columns_path, 'r') as fh:
            columns = json.load(fh)

        return booster, columns

# ------------------------------ Automatization ------------------------------
class AutoLineups:
    def __init__(self, schedule_id):
        self.schedule_id = schedule_id

        sql_query = f"""
            SELECT 
                *
            FROM schedule_data
            WHERE schedule_id = '{self.schedule_id}';
        """
        result = DB.select(sql_query)
        home_team_id = int(result["home_team_id"].iloc[0])
        away_team_id = int(result["away_team_id"].iloc[0])
        match_url = result['ss_url'].iloc[0]

        match = re.search(r'id:(\d+)', match_url)
        if not match:
            raise ValueError("Match ID not found in URL.")
        match_id = match.group(1)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.121 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        gresponse = requests.get(api_url, headers=headers)

        if gresponse.status_code != 200:
            raise RuntimeError(f"API request failed with status {gresponse.status_code}")

        event_gdata = gresponse.json()
        self.referee_name = event_gdata["event"]["referee"]["name"]

        api_lineups_url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
        lresponse = requests.get(api_lineups_url, headers=headers)

        if lresponse.status_code != 200:
            raise RuntimeError(f"API request failed with status {lresponse.status_code}")
        
        event_ldata = lresponse.json()

        if not event_ldata.get("confirmed", False):
            print("Lineups are NOT confirmed. Exiting.")
            return

        lineups = {
            "home": {"starters": [], "bench": []},
            "away": {"starters": [], "bench": []},
        }

        for side in ["home", "away"]:
            team_data = event_ldata.get(side, {})
            for player_info in team_data.get("players", []):
                player_name = player_info.get("player", {}).get("name", "Unknown")
                if player_info.get("substitute", False):
                    lineups[side]["bench"].append(player_name)
                else:
                    lineups[side]["starters"].append(player_name)

        self.home_starters = lineups["home"]["starters"]
        self.home_subs = lineups["home"]["bench"]
        self.away_starters = lineups["away"]["starters"]
        self.away_subs = lineups["away"]["bench"]

        home_ids_st, home_ids_bn = match_players(home_team_id, self.home_starters + self.home_subs)
        away_ids_st, away_ids_bn = match_players(away_team_id, self.away_starters + self.away_subs)

        def _build_dicts(starters, bench):
            data = []
            for pid in starters:
                data.append(
                    dict(player_id=pid,
                        yellow_card=False,
                        red_card=False,
                        on_field=True,
                        bench=False)
                )
            for pid in bench:
                data.append(
                    dict(player_id=pid,
                        yellow_card=False,
                        red_card=False,
                        on_field=False,
                        bench=True)
                )
            return data
        
        self.home_players_data = _build_dicts(home_ids_st, home_ids_bn)
        self.away_players_data = _build_dicts(away_ids_st, away_ids_bn)

        send_lineup_to_db(self.home_players_data, schedule_id=self.schedule_id, team="home")
        send_lineup_to_db(self.away_players_data, schedule_id=self.schedule_id, team="away")

        sql_query = f"UPDATE schedule_data SET referee_name = %s WHERE schedule_id = %s"
        DB.execute(sql_query, (self.referee_name, schedule_id))

class AutoMatchInfo:
    def __init__(self, schedule_id):
        self.schedule_id = schedule_id

        sql_query = f"""
            SELECT 
                *
            FROM schedule_data
            WHERE schedule_id = '{self.schedule_id}';
        """
        result = DB.select(sql_query)
        match_url = result['ss_url'].iloc[0]
        home_players_data = result['home_players_data'].iloc[0]
        away_players_data = result['away_players_data'].iloc[0]
        last_minute_checked = int(result['last_minute_checked'].iloc[0] or 0)

        match = re.search(r'id:(\d+)', match_url)
        if not match:
            raise ValueError("Match ID not found in URL.")
        match_id = match.group(1)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.121 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        gresponse = requests.get(api_url, headers=headers)

        if gresponse.status_code != 200:
            raise RuntimeError(f"API request failed with status {gresponse.status_code}")

        event_gdata = gresponse.json()
        self.home_score = int(event_gdata["event"]["homeScore"]["current"])
        self.away_score = int(event_gdata["event"]["awayScore"]["current"])

        self.current_period_start_timestamp = int(event_gdata["event"]["time"]["currentPeriodStartTimestamp"])
        self.period = event_gdata.get("event", {}).get("lastPeriod")
        if self.period and self.period[-1].isdigit():
            injury_key = f"injuryTime{self.period[-1]}"
        else:
            injury_key = None
        self.period_injury_time = int(event_gdata["event"]["time"].get(injury_key)) if injury_key and event_gdata["event"]["time"].get(injury_key) is not None else None

        def _clean(txt: str) -> str:
            return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode().lower()

        def _match_player_id(api_name: str, squad_names: dict[str, str]) -> str | None:
            api_name = _clean(api_name)

            if api_name in squad_names:
                return squad_names[api_name]

            best = process.extractOne(api_name, squad_names.keys(), score_cutoff=70)
            return squad_names[best[0]] if best else None

        def parse_incidents(
            incidents: list[dict],
            home_status: list[dict],
            away_status: list[dict],
            last_minute_checked: int = 0
        ) -> tuple[dict, list[dict], list[dict], int, int, int]:
            events = {
                "home": {"substitutions": [], "yellow_cards": [], "red_cards": []},
                "away": {"substitutions": [], "yellow_cards": [], "red_cards": []},
            }

            def _build_idx(players_status):
                return {
                    _clean(p["player_id"].split("_")[0]): p["player_id"]
                    for p in players_status
                }

            idx_home, idx_away = _build_idx(home_status), _build_idx(away_status)

            last_minute = last_minute_checked
            cnt = {
                "home": {"sub": 0, "yellow": 0, "red": 0},
                "away": {"sub": 0, "yellow": 0, "red": 0},
            }

            # ------------------------------------------------------------------------    
            for inc in incidents:
                minute     = inc.get("time", 0)
                if minute <= last_minute_checked:
                    continue

                side       = "home" if inc.get("isHome") else "away"
                squad_idx  = idx_home if side == "home" else idx_away
                squad_stat = home_status if side == "home" else away_status

                inc_type   = inc.get("incidentType")
                last_minute = max(last_minute, minute)

                # substitutions -------------------------------------------------------
                if inc_type == "substitution":
                    cnt[side]["sub"] += 1
                    pid_in  = _match_player_id(inc["playerIn"]["name"],  squad_idx)
                    pid_out = _match_player_id(inc["playerOut"]["name"], squad_idx)

                    for pl in squad_stat:
                        if pl["player_id"] == pid_in:
                            pl.update({"bench": False, "on_field": True})
                        elif pl["player_id"] == pid_out:
                            pl.update({"bench": False, "on_field": False})

                    events[side]["substitutions"].append(
                        {"minute": minute, "in": pid_in, "out": pid_out}
                    )

                # cards ----------------------------------------------------------------
                elif inc_type == "card":
                    pid = _match_player_id(inc["player"]["name"], squad_idx)

                    if inc["incidentClass"] == "yellow":
                        cnt[side]["yellow"] += 1
                        events[side]["yellow_cards"].append({"minute": minute, "player": pid})
                        for pl in squad_stat:
                            if pl["player_id"] == pid:
                                pl["yellow_card"] = True

                    elif inc["incidentClass"] == "red":
                        cnt[side]["red"] += 1
                        events[side]["red_cards"].append({"minute": minute, "player": pid})
                        for pl in squad_stat:
                            if pl["player_id"] == pid:
                                pl["red_card"] = True

            # simulate flags ----------------------------------------------------------
            simulate_home = int(
                cnt["home"]["sub"] > 0 or cnt["home"]["red"] > 0 or cnt["home"]["yellow"] >= 2
            )
            simulate_away = int(
                cnt["away"]["sub"] > 0 or cnt["away"]["red"] > 0 or cnt["away"]["yellow"] >= 2
            )

            return (
                home_status,
                away_status,
                last_minute,
                simulate_home,
                simulate_away,
            )

        api_incidents_url = f"https://www.sofascore.com/api/v1/event/{match_id}/incidents"
        iresponse = requests.get(api_incidents_url, headers=headers)

        incidents_data = iresponse.json()["incidents"]

        upd_home, upd_away, last_min, sim_home, sim_away = parse_incidents(
            incidents_data,
            json.loads(home_players_data),
            json.loads(away_players_data),
            last_minute_checked  
        )

        simulate = int(sim_home or sim_away)

        DB.execute(
            """
            UPDATE schedule_data
            SET home_players_data               = %s,
                away_players_data              = %s,
                last_minute_checked            = %s,
                simulate                       = %s,
                current_home_goals             = %s,
                current_away_goals             = %s,
                current_period_start_timestamp = %s,
                period                         = %s,
                period_injury_time             = %s
            WHERE schedule_id = %s;
            """,
            (
                json.dumps(upd_home),
                json.dumps(upd_away),
                last_min,
                simulate,
                self.home_score,
                self.away_score,
                self.current_period_start_timestamp,
                self.period,
                self.period_injury_time,
                self.schedule_id
            )
        )

# ------------------------------ Trading ------------------------------
class MatchTrade:
    def __init__(self, matched_bets):
        self.matched_bets = matched_bets
        self.selections_pl = self.profit_loss(self.matched_bets)

    def profit_loss(self, matched_bets):
        selections = {"Home": 0, "Away": 0, "Draw": 0}
        for bet in matched_bets:
            if bet["Type"] == "Back":
                bet_profit = bet["Amount"]*(bet["Odds"]-1)
                bet_liability = bet["Amount"]
            else:
                bet_profit = bet["Amount"]
                bet_liability = bet["Amount"]*(bet["Odds"]-1)
            bet["Profit"] = bet_profit
            bet["Liability"] = bet_liability

        for selection in selections.keys():
            pl = 0
            for bet in matched_bets:
                if selection == bet["Selection"]:
                    if bet["Type"] == "Back":
                        pl += bet["Profit"]
                    else:
                        pl -= bet["Liability"]
                else:
                    if bet["Type"] == "Back":
                        pl -= bet["Liability"]
                    else:
                        pl += bet["Profit"]

            selections[selection] = pl
        return selections

class TWTrade:  
    def __init__(self, matched_bets):
        self.matched_bets = matched_bets
        self.selections_pl = self.profit_loss(self.matched_bets)

    def profit_loss(self, matched_bets):
        if matched_bets and matched_bets[0]["Selection"] in ["Home AH", "Away AH"]:
            outcomes = ["Home AH", "Away AH"]
        else:
            outcomes = ["Over", "Under"]
        selections = {outcome: 0 for outcome in outcomes}
        for bet in matched_bets:
            if bet["Type"] == "Back":
                bet_profit = bet["Amount"] * (bet["Odds"] - 1)
                bet_liability = bet["Amount"]
            else:
                bet_profit = bet["Amount"]
                bet_liability = bet["Amount"] * (bet["Odds"] - 1)
            bet["Profit"] = bet_profit
            bet["Liability"] = bet_liability
        for outcome in outcomes:
            pl = 0
            for bet in matched_bets:
                if bet["Selection"] == outcome:
                    if bet["Type"] == "Back":
                        pl += bet["Profit"]
                    else:
                        pl -= bet["Liability"]
                else:
                    if bet["Type"] == "Back":
                        pl -= bet["Liability"]
                    else:
                        pl += bet["Profit"]
            selections[outcome] = pl
        return selections   

class ScoreTrade:
    def __init__(self, matched_bets):
        self.matched_bets = matched_bets
        self.selections_pl = self.profit_loss(self.matched_bets)

    def profit_loss(self, matched_bets):
        selections = {key: 0 for key in ["0-0", "0-1", "0-2", "0-3", "1-0", "1-1", "1-2", "1-3", "2-0", "2-1", "2-2", "2-3", "3-0", "3-1", "3-2", "3-3", "Home Win 4+", "Away Win 4+", "Draw +4"]}
        for bet in matched_bets:
            if bet["Type"] == "Back":
                bet_profit = bet["Amount"]*(bet["Odds"]-1)
                bet_liability = bet["Amount"]
            else:
                bet_profit = bet["Amount"]
                bet_liability = bet["Amount"]*(bet["Odds"]-1)
            bet["Profit"] = bet_profit
            bet["Liability"] = bet_liability

        for selection in selections.keys():
            pl = 0
            for bet in matched_bets:
                if selection == bet["Selection"]:
                    if bet["Type"] == "Back":
                        pl += bet["Profit"]
                    else:
                        pl -= bet["Liability"]
                else:
                    if bet["Type"] == "Back":
                        pl -= bet["Liability"]
                    else:
                        pl += bet["Profit"]

            selections[selection] = pl
        return selections
    

    def dutching(self, total_stake, selections_odds):
        stakes = {}
        total_inverse_odds = sum(1/odds for odds in selections_odds.values())
        for selection, odds in selections_odds.items():
            stakes[selection] = (total_stake / total_inverse_odds) / odds
        return stakes