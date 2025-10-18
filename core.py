from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable, Sequence
import pandas as pd
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime, date, timedelta
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
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from scipy import sparse
import json
from tqdm import tqdm
import ast
import xgboost as xgb
import multiprocessing
import os
import itertools
import copy
import unicodedata
from dotenv import load_dotenv

# ------------------------------ Database ------------------------------
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
    def __init__(self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        pool_name: str = "db_pool",
        pool_size: int = 6,
    ) -> None:
        self._pool: MySQLConnectionPool = MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            host=host,
            port=port,
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

load_dotenv()
host = os.getenv('DB_HOST')
port = int(os.getenv('DB_PORT'))
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')

try:
    DB = DatabaseManager(host="localhost", user=user, password="venomio", database="vpfm")
except Exception as e:
    print(f"[INFO] Could not connect to DB: {e}")
    DB = None

# --------------- Useful Functions & Variables ---------------
def get_team_name_by_id(id: int):
    query = "SELECT name FROM teams WHERE id = %s"
    result = DB.select(query, (id,))
    if not result.empty:
        return result.iloc[0]["name"]
    return None

def get_team_id_by_name(team_name: str, league_id: int):
    query = "SELECT id FROM teams WHERE name = %s AND league_id = %s"
    result = DB.select(query, (team_name, league_id))
    if not result.empty:
        return int(result.iloc[0]["id"])
    return None

def get_league_name_by_id(league_id: int):
    query = "SELECT name FROM leagues WHERE id = %s"
    result = DB.select(query, (league_id,))
    if not result.empty:
        return result.iloc[0]["name"]
    return None

def match_players(team_id, raw_source):
    def _normalize(text: str) -> str:
        """
        1. Quita acentos.
        2. Elimina todo lo que no sea letra o espacio.
        3. Convierte a minúsculas y colapsa espacios.
        """
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return " ".join(text.lower().split())

    if hasattr(raw_source, "toPlainText"):               
        raw_text = raw_source.toPlainText()
        raw_list = [line.strip() for line in raw_text.split("\n")]
    elif isinstance(raw_source, (list, tuple)):         
        raw_list = [str(line).strip() for line in raw_source]
    else:
        raise TypeError("raw_source must be QTextEdit-like or list/tuple")

    clean_list = [l for l in raw_list if l and not any(c.isdigit() for c in l)]

    unmatched_starters = clean_list[:11]
    unmatched_benchers = clean_list[11:]

    player_sql_query = """
        SELECT  player_id,
                SUBSTRING_INDEX(player_id, '_', 1) AS player_name
        FROM    players_data
        WHERE   current_team = %s;
    """
    players_df = DB.select(player_sql_query, (team_id,))

    id_to_name = {row.player_id: _normalize(row.player_name)
                  for row in players_df.itertuples(index=False)}
    remaining_ids, remaining_names = list(id_to_name.keys()), list(id_to_name.values())

    def _pick_id(raw_name: str, thr: int):
        norm = _normalize(raw_name)
        match = process.extractOne(norm, remaining_names,
                                   scorer=fuzz.token_set_ratio,
                                   score_cutoff=thr)
        if not match:
            return None
        idx = remaining_names.index(match[0])
        remaining_names.pop(idx) 
        return remaining_ids.pop(idx)

    def _resolve(names):
        out, pending, thr = [], names, 85
        while pending and thr >= 60: 
            still = []
            for n in pending:
                pid = _pick_id(n, thr)
                (out if pid else still).append(pid or n)
            pending, thr = still, thr - 10
        return out

    matched_starters = _resolve(unmatched_starters)
    matched_benchers = _resolve(unmatched_benchers)
    return matched_starters, matched_benchers

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

def get_match_title(id: int):
    sql_query = "SELECT home_team_id, away_team_id FROM schedule WHERE id = %s"
    result = DB.select(sql_query, (id,))

    if not result.empty:
        home_name = get_team_name_by_id(int(result.iloc[0]["home_team_id"]))
        away_name = get_team_name_by_id(int(result.iloc[0]["away_team_id"]))

        return f"{home_name} vs {away_name}"  
    else:
        return id

def flip(series: pd.Series) -> pd.Series:
    flipped = -series
    flipped[series == 0] = 0.0
    return flipped

_ID_RE = re.compile(r"(?P<name>.+?)_\d+_[A-Z]{1,2}$")
# ------------------------------ Fetch & Remove Data ------------------------------
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

class UpdateSchedule:
    def __init__(self, from_date):
        self.from_date = from_date

    def update_all_leagues(self):
        active_leagues_df = DB.select("SELECT * FROM league_data WHERE is_active = 1")
        
        for league_id in tqdm(active_leagues_df["league_id"].tolist(), desc="Processing leagues"):
            weather_missing_rows = DB.select(
                """
                SELECT schedule_id
                FROM schedule_data
                WHERE (temperature IS NULL OR is_raining IS NULL)
                AND date < %s
                AND league_id = %s
                """,
                (self.from_date, league_id)
            )

            for sid in weather_missing_rows["schedule_id"].tolist():
                self.update_game_weather(sid)
            
            fbref_url = active_leagues_df[active_leagues_df['league_id'] == league_id]['fbref_fixtures_url'].values[0]
            upto_date = self.from_date + timedelta(days=5)

            games_dates, games_local_time, games_venue_time, home_teams, away_teams = self.get_games_basic_info(fbref_url, upto_date)

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
                home_id = get_team_id_by_name(home_team, league_id)
                away_team = away_teams[i]
                away_id = get_team_id_by_name(away_team, league_id)

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
                    away_rest_days
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    date         = VALUES(date),
                    local_time   = VALUES(local_time),
                    venue_time   = VALUES(venue_time)
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
                    away_rest_days
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

    def update_game_weather(self, schedule_id):
        row = DB.select(
            "SELECT home_team_id, away_team_id, date, venue_time, league_id FROM schedule_data WHERE schedule_id = %s",
            (schedule_id,)
        )
        if row.empty:
            return

        home_id = row["home_team_id"].iat[0]
        game_date = row["date"].iat[0]
        game_venue_time = row["venue_time"].iat[0]

        temp, rain = self.get_weather(home_id, game_date, game_venue_time)

        update_sql = """
        UPDATE schedule_data
        SET temperature = %s,
            is_raining  = %s
        WHERE schedule_id = %s
        """
        DB.execute(update_sql, (temp, rain, schedule_id))

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
            try:
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
            except Exception:
                continue

        driver.quit()
        return games_dates, games_local_time, games_venue_time, home_teams, away_teams

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
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--ignore-certificate-errors")
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

        total_seconds = game_venue_time.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        dummy_date = datetime(2000, 1, 1, hours, minutes)
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
            if start_datetime.time() <= dt.time() <= end_datetime.time() and temp is not None:
                filtered_temps.append(temp)
                filtered_rains.append(rain)

        if not filtered_temps:
            return None, None

        if filtered_temps:
            avg_temp = sum(filtered_temps) / len(filtered_temps)
        else:
            avg_temp = None
        raining = any(r > 0.0 for r in filtered_rains)
        return avg_temp, raining

class ScrapeMatchesData:
    """
    """
    def __init__(self, upto_date: datetime = datetime.now()):
        self.upto_date = upto_date
        #self.get_recent_games_match_info()
        #self.update_matches_info() 
        self._set_pd_raxg()
        #self.remove_old_data()

    def get_recent_games_match_info(self):
        def get_games_basic_info(url, lud):
            s=Service('chromedriver.exe')
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

            fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.stats_table")))
            rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
            filtered_games_urls = []
            games_dates = []
            game_times = []
            home_teams = []
            away_teams = []
            referees = []

            for row in rows:
                try:
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
                except Exception:
                    continue
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
                home_id = get_team_id_by_name(home_team, league_id)
                away_team = away_teams[i]
                away_id = get_team_id_by_name(away_team, league_id)
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

        for _, mi_row in tqdm(missing_matches_df.iterrows(), total=len(missing_matches_df), desc="Missing Matches"):
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
            
    def _set_pd_raxg(self):
        """
        Before processing new data, update the pre defined RAxG for past matches with the current RAxG coefficients
        """
        non_pd_raxg_matches_query = """
            SELECT 
                md.id,
                md.match_id,
                mg.league_id,
                md.teamA_players,
                md.teamB_players,
                md.minutes_played
            FROM match_detailed md
            JOIN match_general mg
                ON mg.id = md.match_id
            WHERE (md.teamA_pd_raxg IS NULL OR md.teamB_pd_raxg IS NULL)
            AND mg.date <= %s
        """
        non_pd_raxg_matches_df = DB.select(non_pd_raxg_matches_query, (self.upto_date,))

        non_pd_raxg_matches_df['teamA_players'] = non_pd_raxg_matches_df['teamA_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
        non_pd_raxg_matches_df['teamB_players'] = non_pd_raxg_matches_df['teamB_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))

        active_players = set()
        for _, row in non_pd_raxg_matches_df.iterrows():
            active_players.update(row['teamA_players'])
            active_players.update(row['teamB_players'])

        if active_players:
            placeholders = ','.join(['%s'] * len(active_players))
            players_sql = f"""
                SELECT id, off_xg_coef, def_xg_coef
                FROM players
                WHERE id IN ({placeholders});
            """
            active_players_df = DB.select(players_sql, list(active_players))
            off_xg_coef_dict = active_players_df.set_index("id")["off_xg_coef"].to_dict()
            def_xg_coef_dict = active_players_df.set_index("id")["def_xg_coef"].to_dict()
        else:
            off_xg_coef_dict, def_xg_coef_dict = {}, {}

        league_df = DB.select("SELECT id, xg_baseline FROM leagues")
        baseline_dict = league_df.set_index("id")["xg_baseline"].to_dict()

        for _, row in non_pd_raxg_matches_df.iterrows():
            minutes = row['minutes_played']
            league_id = row['league_id']
            baseline = baseline_dict.get(league_id, 0.0)

            teamA_ids = row['teamA_players']
            teamB_ids = row['teamB_players']

            teamA_offense = sum(off_xg_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_defense = sum(def_xg_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_pd_raxg = (baseline + teamA_offense - teamB_defense) * minutes

            teamB_offense = sum(off_xg_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_defense = sum(def_xg_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_pd_raxg = (baseline + teamB_offense - teamA_defense) * minutes

            DB.execute("UPDATE match_detailed SET teamA_pd_raxg = %s, teamB_pd_raxg = %s WHERE id = %s", (teamA_pd_raxg, teamB_pd_raxg, row['id']))

    def remove_old_data(self):
        self.a_year_ago_date = self.upto_date - timedelta(days=365)

        delete_query  = """
        DELETE FROM match_info 
        WHERE date < %s
        """

        DB.execute(delete_query, (self.a_year_ago_date,))

# ------------------------------ Process data ------------------------------
class ProcessData:
    """
    Process players data
    """
    def __init__(self, current_date: datetime = datetime.now()):
        self.current_date = current_date
        DB.execute("TRUNCATE TABLE players")

        steps = [
            ("Inserting players basics and unifying duplicates", self._insert_players_basics),
            ("Updating players raxg coef", self._update_players_raxg_coef),
            ("Updating players totals", self._update_players_totals),
            ("Updating reg totals", self._update_reg_totals),
            ("Training contextual raxg xgboost model", self._train_contextual_raxg_xgb_model)
        ]
        
        for desc, step_func in tqdm(steps, desc="Progress"):
            print(desc)
            step_func()

    def _insert_players_basics(self):
        """
        Inserts players name and current team id. Then runs a function to unify duplicated players.
        """
        sql = """
        SELECT md.teamA_players, md.teamB_players, mg.home_team_id, mg.away_team_id 
        FROM match_detailed md
        JOIN match_general mg
            ON md.match_id = mg.id
        """
        result = DB.select(sql, ())
        
        if result.empty:
            return 0
        
        result['teamA_players'] = result['teamA_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
        result['teamB_players'] = result['teamB_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))

        players_set = set()
        for _, row in result.iterrows():
            teamA_players = row['teamA_players'] if isinstance(row['teamA_players'], list) else [row['teamA_players']]
            teamB_players = row['teamB_players'] if isinstance(row['teamB_players'], list) else [row['teamB_players']]
            home_team = int(row["home_team_id"])
            away_team = int(row["away_team_id"])
        
            for player in teamA_players:
                players_set.add((player, home_team))
            
            for player in teamB_players:
                players_set.add((player, away_team))
        
        insert_sql = "INSERT IGNORE INTO players (id, team_id) VALUES (%s, %s)"
        DB.execute(insert_sql, list(players_set), many=True)

        #self._unify_duplicated_players()

    def _unify_duplicated_players(self):
        """
        This function consolidates duplicate player records by:
        1. Grouping player IDs by their team and name (disregarding jersey number).
        2. Identifying groups where IDs never appear together in the same match—these are considered the same player.
        3. For each group:
            • Selecting the most recent ID as the canonical one.
            • Updating all other database tables to reference this canonical ID.
            • Deleting the obsolete player IDs from the master player table.
        """
        df = DB.select("SELECT id, team_id FROM players")
        groups = {}
        for pid, team in df[["id", "team_id"]].itertuples(index=False):
            m = _ID_RE.match(pid)
            if not m:
                continue
            key = (team, m.group("name").strip().lower())
            groups.setdefault(key, []).append(pid)
        
        for (team, _), ids in tqdm(groups.items(), desc="Group of players to unify:"):
            if len(ids) < 2:
                continue
            
            if self._appear_together(ids):
                continue

            keep_id  = self._most_recent_id(ids)

            obsolete = [i for i in ids if i != keep_id]

            self._rewrite_ids(obsolete, keep_id)

            if obsolete:
                ph = ",".join(["%s"] * len(obsolete))
                DB.execute(f"DELETE FROM players WHERE id IN ({ph})", tuple(obsolete),)

    def _appear_together(self, ids):
        ors = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ors.append(
                    "( (JSON_CONTAINS(teamA_players, '\"%s\"') AND "
                    "   JSON_CONTAINS(teamA_players, '\"%s\"')) OR "
                    "  (JSON_CONTAINS(teamB_players, '\"%s\"') AND "
                    "   JSON_CONTAINS(teamB_players, '\"%s\"')) )" % (a, b, a, b)
                )
        sql = "SELECT 1 FROM match_detailed WHERE " + " OR ".join(ors) + " LIMIT 1"
        return not DB.select(sql, ()).empty

    def _most_recent_id(self, ids):
        most_recent   = None
        latest_game   = -1

        for pid in ids:
            sql = """
            SELECT MAX(match_id) AS last_game
            FROM   match_detailed
            WHERE  JSON_CONTAINS(teamA_players, JSON_QUOTE(%s), '$')
               OR  JSON_CONTAINS(teamB_players, JSON_QUOTE(%s), '$')
            """
            res = DB.select(sql, (pid, pid))
            last = res.iloc[0]["last_game"] if not res.empty else None
            if last is not None and last > latest_game:
                latest_game = last
                most_recent = pid

        return most_recent or sorted(ids)[-1]

    def _rewrite_ids(self, olds, new):
        # ---------- match_detailed ----------
        cond = " OR ".join(["JSON_CONTAINS(teamA_players,'\"%s\"') OR JSON_CONTAINS(teamB_players,'\"%s\"')" % (o, o)for o in olds])
        rows = DB.select(f"SELECT match_id, teamA_players, teamB_players FROM match_detailed WHERE {cond}", ())

        for _, r in rows.iterrows():
            a = json.dumps([new if p in olds else p for p in json.loads(r["teamA_players"])])
            b = json.dumps([new if p in olds else p for p in json.loads(r["teamB_players"])])
            DB.execute("UPDATE match_detailed SET teamA_players=%s, teamB_players=%s WHERE match_id=%s",(a, b, r["match_id"]),)

        # ---------- match_player_breakdown ----------
        for o in olds:
            DB.execute("UPDATE IGNORE match_player_breakdown ""SET player_id = %s ""WHERE player_id = %s",(new, o),)
            DB.execute("DELETE FROM match_player_breakdown WHERE player_id = %s",(o,),)

    def _update_players_raxg_coef(self):
        """
        A ridge regression (linear model) that learns individual players offensive and defensive impact. 
        Having more weight on recent matches, for up to matches for the preceding year.
        It updates players coefficients per league
        """
        leagues_df = DB.select("SELECT id FROM leagues WHERE is_active = 1")

        for lid in tqdm(leagues_df['id'].tolist(), desc="League RAxG Coeff"):
            matches_query = """
            SELECT 
                mg.id, 
                mg.date,
                md.teamA_players, 
                md.teamB_players, 
                md.teamA_xg,
                md.teamB_xg,
                md.minutes_played
            FROM match_general mg
            LEFT JOIN match_detailed md
                ON mg.id = md.match_id
            WHERE league_id = %s
            """
            matches_df = DB.select(matches_query, (lid,))

            if matches_df.empty:
                continue 

            matches_df['date'] = pd.to_datetime(matches_df['date'])
            matches_df['days_ago'] = (self.current_date - matches_df['date']).dt.days

            matches_df['teamA_players'] = matches_df['teamA_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
            matches_df['teamB_players'] = matches_df['teamB_players'].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
            matches_df['time_weight'] = np.exp(-np.log(2) * matches_df['days_ago'] / 180)
            total_weight = matches_df['time_weight'].sum()
            matches_df['time_weight'] = matches_df['time_weight'] / total_weight * len(matches_df)

            players_set = set()
            for idx, row in matches_df.iterrows():
                teamA = row['teamA_players'] if isinstance(row['teamA_players'], list) else [row['teamA_players']]
                teamB = row['teamB_players'] if isinstance(row['teamB_players'], list) else [row['teamB_players']]
                players_set.update(teamA)
                players_set.update(teamB)

            players = sorted(list(players_set))
            num_players = len(players)
            players_to_index = {player: idx for idx, player in enumerate(players)}

            rows = []
            cols = []
            data_vals = []
            y = []
            sample_weights = []
            row_num = 0

            for idx, row in matches_df.iterrows():
                minutes = int(row['minutes_played'])
                if minutes == 0:
                    continue
                
                time_weight = float(row['time_weight'])
                teamA_players = row['teamA_players'] if isinstance(row['teamA_players'], list) else [row['teamA_players']]
                teamB_players = row['teamB_players'] if isinstance(row['teamB_players'], list) else [row['teamB_players']]
                teamA_xg = float(row['teamA_xg'])
                teamB_xg = float(row['teamB_xg'])

                # Team A's offensive possessions (positive for offense, negative for defense)
                for p in teamA_players:
                    rows.append(row_num)
                    cols.append(players_to_index[p])
                    data_vals.append(1)
                for p in teamB_players:
                    rows.append(row_num)
                    cols.append(num_players + players_to_index[p])
                    data_vals.append(-1)
                y.append(teamA_xg / minutes)
                sample_weights.append(np.sqrt(minutes * time_weight))
                row_num += 1

                # Team B's offensive possessions
                for p in teamB_players:
                    rows.append(row_num)
                    cols.append(players_to_index[p])
                    data_vals.append(1)
                for p in teamA_players:
                    rows.append(row_num)
                    cols.append(num_players + players_to_index[p])
                    data_vals.append(-1)
                y.append(teamB_xg / minutes)
                sample_weights.append(np.sqrt(minutes * time_weight))
                row_num += 1

            X = sparse.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
            y_array = np.array(y)
            sample_weights_array = np.array(sample_weights)

            y_mean = np.average(y_array, weights=sample_weights_array)
            y_centered = y_array - y_mean


            alphas = np.logspace(-3, 3, 13)
            ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
            ridge_cv.fit(X, y_centered, sample_weight=sample_weights_array)

            ridge = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
            ridge.fit(X, y_centered, sample_weight=sample_weights_array)

            intercept_adjusted = ridge.intercept_ + y_mean

            offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
            defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))

            for player in players:
                off_xg_coef = offensive_ratings[player]
                def_xg_coef = defensive_ratings[player]
                update_coef_query = """
                UPDATE players
                SET off_xg_coef = %s, def_xg_coef = %s
                WHERE id = %s
                """
                DB.execute(update_coef_query, (float(off_xg_coef), float(def_xg_coef), player))

            DB.execute("UPDATE leagues SET xg_baseline = %s WHERE id = %s", (float(intercept_adjusted), lid))

    def _update_players_totals(self):
        """
        Sums all data from all players
        """
        players_df = DB.select("SELECT DISTINCT id FROM players")
        
        for pid in tqdm(players_df["id"].tolist(), desc="Total Players"):
            reg_query = """
            SELECT
                COALESCE(SUM(minutes_played), 0) AS minutes_played,
                COALESCE(SUM(fouls_committed), 0) AS fouls_committed,
                COALESCE(SUM(fouls_drawn), 0) AS fouls_drawn,
                COALESCE(SUM(yellow_cards), 0) AS yellow_cards,
                COALESCE(SUM(red_cards), 0) AS red_cards
            FROM match_player_breakdown
            WHERE player_id = %s
            """
            reg_result = DB.select(reg_query, (pid,))

            if reg_result.empty:
                continue

            player_row = reg_result.iloc[0]

            pt_query = """
            SELECT in_status, out_status, sub_in, sub_out
            FROM match_player_breakdown
            WHERE player_id = %s
            """
            pt_result = DB.select(pt_query, (pid,))
            
            in_status_dict = {"trailing": 0, "level": 0, "leading": 0}
            out_status_dict = {"trailing": 0, "level": 0, "leading": 0}
            subs_in_list = []
            subs_out_list = []
            
            for _, pt_row in pt_result.iterrows():
                in_stat = pt_row["in_status"]
                out_stat = pt_row["out_status"]

                if in_stat in in_status_dict:
                    in_status_dict[in_stat] += 1
                if out_stat in out_status_dict:
                    out_status_dict[out_stat] += 1

                sub_in_val = pt_row["sub_in"]
                if pd.notna(sub_in_val) and sub_in_val != "":
                    subs_in_list.append(sub_in_val)

                sub_out_val = pt_row["sub_out"]
                if pd.notna(sub_out_val) and sub_out_val != "":
                    subs_out_list.append(sub_out_val)

            player_update_query = """
            UPDATE players
            SET 
                minutes_played = %s,
                fouls_committed = %s,
                fouls_drawn = %s,
                yellow_cards = %s,
                red_cards = %s,
                in_status = %s,
                out_status = %s,
                sub_in = %s,
                sub_out = %s
            WHERE id = %s
            """
            DB.execute(player_update_query, (
                int(player_row["minutes_played"]),
                int(player_row["fouls_committed"]),
                int(player_row["fouls_drawn"]),
                int(player_row["yellow_cards"]),
                int(player_row["red_cards"]),
                json.dumps(in_status_dict),
                json.dumps(out_status_dict),
                json.dumps(subs_in_list, allow_nan=False),
                json.dumps(subs_out_list, allow_nan=False),
                pid
            ))

    def _update_reg_totals(self):
        """
        Updates regularization data in leagues
        """
        update_sql = """
        WITH reg AS (
        SELECT 
            mpb.match_id,
            mg.league_id,
            COALESCE(SUM(fouls_committed),0) AS total_fouls,
            COALESCE(SUM(yellow_cards),0)    AS yellow_cards,
            COALESCE(SUM(red_cards),0)       AS red_cards
        FROM match_player_breakdown mpb
        LEFT JOIN match_general mg
            ON mpb.match_id = mg.id
        GROUP BY match_id
        ), league_totals AS (
        SELECT
            league_id,
            COUNT(*) AS total_matches,
            SUM(total_fouls) AS total_fouls,
            SUM(yellow_cards) AS yellow_cards,
            SUM(red_cards) AS red_cards
        FROM reg
        GROUP BY league_id
        )
        UPDATE leagues l
        JOIN league_totals lt
            ON l.id = lt.league_id
        SET
            foul_rate = ROUND(total_fouls/total_matches, 1),
            yc_rate = ROUND(yellow_cards/total_matches, 1),
            rc_rate = ROUND(red_cards/total_matches, 1)
        """
        DB.execute(update_sql)

    def _train_contextual_raxg_xgb_model(self):
        """
        Train a contextual expected goals (xG) XGBOOST model using regularized adjusted xG (RAxG) as baseline.
        
        This model incorporates contextual factors like elevation differences, travel distance,
        match state, and player quality differences to predict team xG performance. The model
        uses Poisson regression with RAxG as base margin and applies sample weighting based
        on minutes played.
        """
        # DELETE THE SUMS AND THE GROUP WHEN THATS FIXED ON THE DAT SCRAPING SECTION
        data_query = """ 
        SELECT 
            mg.id,
            mg.home_elevation_dif,
            mg.away_elevation_dif,
            mg.away_travel,
            mg.date,
            SUM(md.teamA_pd_raxg) AS teamA_pd_raxg,
            SUM(md.teamB_pd_raxg) AS teamB_pd_raxg,
            SUM(md.minutes_played) AS minutes_played,
            md.match_state,
            md.player_dif,
            SUM(md.teamA_xg) AS teamA_xg,
            SUM(md.teamB_xg) AS teamB_xg
        FROM match_general mg
        LEFT JOIN match_detailed md
            ON mg.id = md.match_id
        GROUP BY mg.id, md.match_state, md.player_dif
        """
        context_df = DB.select(data_query)
        context_df = context_df.replace([np.inf, -np.inf], np.nan).dropna()
        context_df['home_elevation_dif']  = pd.to_numeric(context_df['home_elevation_dif'],  errors='raise').astype(int)
        context_df['away_elevation_dif']  = pd.to_numeric(context_df['away_elevation_dif'],  errors='raise').astype(int)
        context_df['away_travel']  = pd.to_numeric(context_df['away_travel'],  errors='raise').astype(int)
        context_df['date'] = pd.to_datetime(context_df['date'])
        context_df['teamA_pd_raxg']  = pd.to_numeric(context_df['teamA_pd_raxg'],  errors='raise').astype(float)
        context_df['teamB_pd_raxg']  = pd.to_numeric(context_df['teamB_pd_raxg'],  errors='raise').astype(float)
        context_df['minutes_played']  = pd.to_numeric(context_df['minutes_played'],  errors='raise').astype(int)
        context_df['match_state'] = pd.to_numeric(context_df['match_state'], errors='raise').astype(float)
        context_df['player_dif']  = pd.to_numeric(context_df['player_dif'],  errors='raise').astype(float)
        context_df['teamA_xg']  = pd.to_numeric(context_df['teamA_xg'],  errors='raise').astype(float)
        context_df['teamB_xg']  = pd.to_numeric(context_df['teamB_xg'],  errors='raise').astype(float)


        home_df = pd.DataFrame({
            'xg'                 : context_df['teamA_xg'],
            'pd_raxg'            : context_df['teamA_pd_raxg'],
            'minutes_played'     : context_df['minutes_played'],
            'is_home'            : 1,
            'elevation_dif'      : context_df['home_elevation_dif'],
            'travel'             : -context_df['away_travel'],
            'match_state'        : context_df['match_state'],
            'player_dif'         : context_df['player_dif']
        })

        away_df = pd.DataFrame({
            'xg'                 : context_df['teamB_xg'],
            'pd_raxg'            : context_df['teamB_pd_raxg'],
            'minutes_played'     : context_df['minutes_played'],
            'is_home'            : 0,
            'elevation_dif'      : context_df['away_elevation_dif'],
            'travel'             : context_df['away_travel'],
            'match_state'        : flip(context_df['match_state']),
            'player_dif'         : flip(context_df['player_dif'])
        })
        
        concat_df = pd.concat([home_df, away_df], ignore_index=True)
        clean_df = concat_df.copy()
        clean_df = clean_df[clean_df['minutes_played'] >= 10]

        clean_df['xg90'] = (clean_df['xg'] / clean_df['minutes_played']) * 90
        clean_df['raxg90'] = (clean_df['pd_raxg'] / clean_df['minutes_played']) * 90

        cat_cols  = ['match_state', 'player_dif']
        bool_cols = ['is_home']
        num_cols  = ['elevation_dif', 'travel']
      
        missing_cols  = [c for c in ['xg90', 'raxg90'] if c not in clean_df.columns]
        if missing_cols:
            raise ValueError(f'Missing expected columns: {missing_cols}')
        
        sample_weights = np.sqrt(clean_df['minutes_played'] / clean_df['minutes_played'].max())

        for c in cat_cols:
            clean_df[c] = clean_df[c].astype(str).str.lower()

        clean_df[bool_cols] = clean_df[bool_cols].astype(int)

        X_cat = pd.get_dummies(clean_df[cat_cols], prefix=cat_cols)
        X = pd.concat([clean_df[num_cols], clean_df[bool_cols], X_cat], axis=1)

        y = clean_df['xg90']
        base_margin = np.log(clean_df['raxg90'].clip(lower=0.01))

        dtrain = xgb.DMatrix(X, label=y, base_margin=base_margin, weight=sample_weights)

        params = dict(objective='count:poisson',
                        tree_method='hist',
                        max_depth=4,
                        eta=0.03,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        min_child_weight=15,
                        gamma=1,
                        reg_alpha=0.5,
                        reg_lambda=1.5,
                        max_delta_step=1)
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            early_stopping_rounds=50,
            metrics='poisson-nloglik',
            verbose_eval=False
        )
        
        optimal_rounds = len(cv_results)
        booster = xgb.train(params, dtrain, num_boost_round=optimal_rounds)

        booster.save_model('Database/craxg_booster.json')
        with open('Database/craxg_columns.json', 'w') as fh:
            json.dump(X.columns.tolist(), fh)

# ------------------------------ Simulation ------------------------------
class MonteCarloSim:
    """
    Match simulation
    """
    def __init__(self, match_id, home_initial_goals=0, away_initial_goals=0, match_initial_time=0, home_subs_avail=5, away_subs_avail=5):
        self.match_id = match_id

        print(f"Simulating {get_match_title(self.match_id)}")
        
        match_df = DB.select("SELECT * FROM schedule WHERE id = %s", (self.match_id,))

        self.home_team_id = int(match_df.iloc[0]['home_team_id'])
        self.away_team_id = int(match_df.iloc[0]['away_team_id'])
        self.match_date = pd.to_datetime(match_df.iloc[0]['date'])
        home_data_raw = match_df.iloc[0]['home_players_data']
        away_data_raw = match_df.iloc[0]['away_players_data']
        if home_data_raw is None or away_data_raw is None:
            raise ValueError("Players data is None")
        self.home_players_init_data = json.loads(home_data_raw)
        self.away_players_init_data = json.loads(away_data_raw)
        self.league_id = int(match_df.iloc[0]['league_id'])
        self.home_elevation_dif = int(match_df.iloc[0]['home_elevation_dif'])
        self.away_elevation_dif = int(match_df.iloc[0]['away_elevation_dif'])
        self.away_travel = int(match_df.iloc[0]['away_travel'])
        self.home_initial_goals = home_initial_goals
        self.away_initial_goals = away_initial_goals
        self.match_initial_time = match_initial_time
        self.home_subs_avail = home_subs_avail
        self.away_subs_avail = away_subs_avail

        xg_baseline_df = DB.select("SELECT xg_baseline FROM leagues WHERE id = %s", (self.league_id,))
        self.xg_baseline_coef = float(xg_baseline_df.iloc[0]['xg_baseline']) if not xg_baseline_df.empty and xg_baseline_df.iloc[0]['xg_baseline'] is not None else 0.0      

        self.craxg_booster, self.craxg_columns = self._load_craxg_model()
        self.craxg_home_multipliers, self.craxg_away_multipliers = self._precompute_craxg_multipliers()

        self.league_regulation_data = self._get_league_regulation_data()

        self.home_starters, self.home_subs = self._divide_players("home")
        self.away_starters, self.away_subs = self._divide_players("away")

        self.home_players_data = self._get_players_data("home")
        self.away_players_data = self._get_players_data("away")

        self.home_sub_minutes, self.away_sub_minutes = self._get_sub_minutes()
        self.all_sub_minutes = list(set(list(self.home_sub_minutes.keys()) + list(self.away_sub_minutes.keys())))

        if self.match_initial_time >= 75:
            range_value = 1000
        elif self.match_initial_time > 60:
            range_value = 2000
        elif self.match_initial_time > 45:
            range_value = 4000
        elif self.match_initial_time > 15:
            range_value = 6000
        elif self.match_initial_time <= 1:
            range_value = 10000
        else:
            range_value = 8000

        self._run_simulations(n_sims=1, n_workers=1, flush_every=10000) # 1 & 1(4) & 10k (1k) for testing

    def _load_craxg_model(self) -> tuple[xgb.Booster, list[str]]:
        booster_path  = f'Database/craxg_booster.json'
        columns_path  = f'Database/craxg_columns.json'

        booster = xgb.Booster()
        booster.load_model(booster_path)

        with open(columns_path, 'r') as fh:
            columns = json.load(fh)

        return booster, columns

    def _simulate_single(self, i: int) -> list:
        """
        Simulates a single game. 
        
        """
        sim_home_players_data = copy.deepcopy(self.home_players_data)
        sim_away_players_data = copy.deepcopy(self.away_players_data)
        print(sim_home_players_data)

        home_goals = self.home_initial_goals
        away_goals = self.away_initial_goals
        home_active_players  = self.home_starters.copy()
        away_active_players  = self.away_starters.copy()
        home_inactive_players = self.home_subs.copy()
        away_inactive_players = self.away_subs.copy()

        score_rows = []

        home_status, away_status = self._get_status(home_goals, away_goals)

        home_raw_raxg = self._get_teams_raw_raxg(0, home_active_players, away_active_players, sim_home_players_data, sim_away_players_data)
        away_raw_raxg = self._get_teams_raw_raxg(0, away_active_players, home_active_players, sim_away_players_data, sim_home_players_data)

        home_mult = self.craxg_home_multipliers[(0, 0)]
        away_mult = self.craxg_away_multipliers[(0, 0)]
        home_context_raxg = max(1e-6, home_raw_raxg) * home_mult
        away_context_raxg = max(1e-6, away_raw_raxg) * away_mult
        
        home_foul_p = self._get_team_foul_prob(home_active_players, away_active_players, home_status, is_home=True)
        away_foul_p = self._get_team_foul_prob(away_active_players, home_active_players, away_status, is_home=False)
        print(home_foul_p)

        context_ras_change = False
        for minute in range(self.match_initial_time, 91):
            home_status, away_status = self._get_status(home_goals, away_goals)
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
                    # print(f"({shooter}, {assister}, home) = {xg_prob} = {outcome}")
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
                    # print(f"({shooter}, {assister}, away) = {xg_prob} = {outcome}")
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
        return score_rows

    def _run_simulations(self, n_sims: int, n_workers: int, flush_every: int = 1000):
        """
        Runs Monte Carlo simulations in parallel (if possible) and saves results to a database in batches to manage memory usage
        """
        if n_workers is None:
            n_workers = os.cpu_count() or 1

        score_buf = []
        first_flush_done = False

        def _flush():
            nonlocal first_flush_done
            if not score_buf:
                return
            self._insert_buf(rows=score_buf, initial_delete=not first_flush_done)
            first_flush_done = True
            score_buf.clear()

        if n_workers > 1:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for idx, score_rows in enumerate(tqdm(pool.imap_unordered(self._simulate_single, range(n_sims)), total=n_sims, desc=f'Simulations ({n_workers} workers)')):
                    score_buf.extend(score_rows)
                    if (idx + 1) % flush_every == 0:
                        _flush()
        else:
            for idx in tqdm(range(n_sims), desc='Simulations (1 worker)'):
                score_rows = self._simulate_single(idx)
                score_buf.extend(score_rows)
                if (idx + 1) % flush_every == 0:
                    _flush()

        _flush()

    def _predict_craxg(self, new_match, *, raw=False):
        match_data = new_match.copy()

        cat_cols  = ['match_state', 'player_dif']
        bool_cols = ['is_home']
        num_cols  = ['elevation_dif', 'travel']
        
        base_margin = np.log(match_data.pop('pd_raxg').clip(lower=0.01))

        for col in cat_cols:
            match_data[col] = match_data[col].astype(str).str.lower()
        
        for col in bool_cols:
            match_data[col] = match_data[col].astype(int)

        new_cat = pd.get_dummies(match_data[cat_cols], prefix=cat_cols)

        new_X = pd.concat([match_data[num_cols + bool_cols].reset_index(drop=True), new_cat.reset_index(drop=True)], axis=1)
        new_X = new_X.reindex(columns=self.craxg_columns, fill_value=0)

        dmatrix = xgb.DMatrix(new_X, base_margin=base_margin)
        prediction = self.craxg_booster.predict(dmatrix, output_margin=raw)

        return prediction[0]

    def _precompute_craxg_multipliers(self):
        """
        Precomputes the predictions of the XG Boost model on RAxG for efficiency
        """
        def _template(is_home: bool):
            return {
                'is_home'       : int(is_home),
                'elevation_dif' : self.home_elevation_dif if is_home else self.away_elevation_dif,
                'travel'        : -self.away_travel if is_home else self.away_travel,
                'pd_raxg'       : 1.0          # 1 ⇒  log(1)=0  ⇒  pure model effect
            }

        states       = [-1.5, -0.5, 0, 0.5, 1.5]
        player_diffs = [-1.5, -0.5, 0, 0.5, 1.5]

        home_cache, away_cache = {}, {}
        for st, pdif in itertools.product(states, player_diffs):
            for is_home, cache in ((True, home_cache), (False, away_cache)):
                row = _template(is_home)
                row.update({'match_state': st, 'player_dif': pdif})
                raw_margin = self._predict_craxg(pd.DataFrame([row]))
                cache[(st, pdif)] = np.exp(raw_margin)
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
 
    def _divide_players(self, team: str) -> tuple[list[str], list[str]]:
        if team == "home":
            players_data = self.home_players_init_data
        elif team == "away":
            players_data = self.away_players_init_data

        starters = [p['player_id'] for p in players_data if p['on_field']]
        subs = [p['player_id'] for p in players_data if p['bench']]
        return starters, subs

    def _get_players_data(self, team: str) -> dict:
        """
        Returns all the neccesary data for each active player
        """
        if team == "home":
            all_players = self.home_starters + self.home_subs
            initial_player_data = self.home_players_init_data
        elif team == "away":
            all_players = self.away_starters + self.away_subs
            initial_player_data = self.away_players_init_data

        escaped_players = [player.replace("'", "''") for player in all_players]
        players_str = ", ".join([f"'{player}'" for player in escaped_players])

        sql_query = f"""
            SELECT 
                *
            FROM players
            WHERE id IN ({players_str});
        """
        players_df = DB.select(sql_query)

        initial_mapping = {}
        if initial_player_data is not None:
            initial_mapping = {player['player_id']: player for player in initial_player_data}

        players_dict = {}
        for player_id in players_df['id'].unique():
            player_row = players_df[players_df['id'] == player_id]
            player_sql_data = player_row.iloc[0].to_dict()

            player_sql_data['fouls_committed_rate'] = self._get_rate(player_sql_data.get('fouls_committed'), player_sql_data.get('minutes_played'))
            player_sql_data['fouls_drawn_rate'] = self._get_rate(player_sql_data.get('fouls_drawn'), player_sql_data.get('minutes_played'))
            player_sql_data['yellow_card_rate'] = self._get_rate(player_sql_data.get('yellow_cards'), player_sql_data.get('fouls_committed'))
            player_sql_data['red_card_rate'] = self._get_rate(player_sql_data.get('red_cards'), player_sql_data.get('fouls_committed'))
            player_sql_data['sub_in_count'] = self._sub_count(player_sql_data.get('sub_in'))
            player_sql_data['sub_out_count'] = self._sub_count(player_sql_data.get('sub_out'))
            player_sql_data['in_status_prob'] = self._status_prob(player_sql_data.get('in_status'))
            player_sql_data['out_status_prob'] = self._status_prob(player_sql_data.get('out_status'))
            player_sql_data['initial_fatigue'], player_sql_data['initial_rhythm'] = self._get_initial_fr(player_sql_data.get('id'))
            player_sql_data['out_status_prob'] = self._status_prob(player_sql_data.get('out_status'))

            delete_columns = ['id', 'team_id', 'fouls_committed', 'fouls_drawn', 'yellow_cards', 'red_cards', 'minutes_played', 'sub_in', 'sub_out', 'in_status', 'out_status']
            player_sql_data = {k: v for k, v in player_sql_data.items() if k not in delete_columns}
            players_dict[player_id] = player_sql_data

            if player_id in initial_mapping:
                extracted = initial_mapping[player_id]
                players_dict[player_id]['bench'] = extracted.get('bench')
                players_dict[player_id]['on_field'] = extracted.get('on_field')
                players_dict[player_id]['yellow_card'] = extracted.get('yellow_card')
                players_dict[player_id]['red_card'] = extracted.get('red_card')
                players_dict[player_id]['in'] = 0 if extracted.get('on_field') else None # Fix this? This should be exctracted only too. so code it elsewehere
            else:
                continue

        return players_dict
    
    def _get_rate(self, numerator: int, denominator: int) -> float:
        if denominator and denominator > 0 and numerator:
            return numerator / denominator
        else:
            return 0.0
        
    def _sub_count(self, raw_str: str) -> float:
        count = 0
        try:
            sub_in_list = eval(raw_str)
            count = len(sub_in_list)
        except:
            pass

        return count
    
    def _status_prob(self, raw_str: str) -> dict:
        counts = {}
        try:
            counts = ast.literal_eval(raw_str)
        except:
            pass

        counts = {k.title(): v for k, v in counts.items()}
        base = {'Leading': 0, 'Level': 0, 'Trailing': 0}
        base.update(counts)

        smoothed = {k: v + 1 for k, v in base.items()}

        total = sum(smoothed.values())
        return {k: v / total for k, v in smoothed.items()}

    def _get_initial_fr(self, player_id: int) -> tuple[float, float]:
        matches_data_query = """
        SELECT
            mpb.minutes_played,
            mg.date
        FROM vpfm.match_player_breakdown mpb
        LEFT JOIN vpfm.match_general mg
            ON mpb.match_id = mg.id
        WHERE mpb.player_id = %s
        """
        matches_data_df = DB.select(matches_data_query, (player_id,))
        matches_data_df['date'] = pd.to_datetime(matches_data_df['date'])

        matches_data_df['days_ago'] = (self.match_date - matches_data_df['date']).dt.days

        # Fatigue calculation (3-day decay)
        total_decayed_minutes_fatigue = sum(minutes * math.exp(-days / 3) for minutes, days in zip(matches_data_df['minutes_played'], matches_data_df['days_ago']))
        initial_fatigue = min(1, total_decayed_minutes_fatigue / 90)

        # Rhythm calculation (14-day decay)
        total_decayed_minutes_rhythm = sum(minutes * math.exp(-days / 14) for minutes, days in zip(matches_data_df['minutes_played'], matches_data_df['days_ago']))
        initial_rhythm = min(1, total_decayed_minutes_rhythm / 90)

        return initial_fatigue, initial_rhythm

    def _get_sub_minutes(self) -> tuple[dict, dict]:
        """
        Returns a dctionary per team for the most common sub windows, and how many subs to do
        """
        sub_minutes_query = """ 
            SELECT 
                mpb.match_id,
                mpb.sub_in,
                p.team_id
            FROM match_player_breakdown mpb
            JOIN players p
                ON mpb.player_id = p.id
            WHERE p.team_id IN (%s, %s)
        """

        sub_minutes_df = DB.select(sub_minutes_query, (self.home_team_id, self.away_team_id))
        sub_minutes_df = sub_minutes_df.dropna()

        home_avg_subs = round(sub_minutes_df[sub_minutes_df['team_id'] == self.home_team_id].groupby('match_id').size().mean())
        away_avg_subs = round(sub_minutes_df[sub_minutes_df['team_id'] == self.away_team_id].groupby('match_id').size().mean())

        effective_home_subs = max(0, min(home_avg_subs - (5 - self.home_subs_avail), self.home_subs_avail))
        effective_away_subs = max(0, min(away_avg_subs - (5 - self.away_subs_avail), self.away_subs_avail))

        home_distribution = self._get_distribution(sub_minutes_df, "home", effective_home_subs)
        away_distribution = self._get_distribution(sub_minutes_df, "away", effective_away_subs)

        return home_distribution, away_distribution

    def _get_distribution(self, df: pd.DataFrame, team: str, effective_subs: int) -> dict:
        if team == "home":
            team_id = self.home_team_id
            avail_subs = self.home_subs_avail
        else:
            team_id = self.away_team_id
            avail_subs = self.away_subs_avail

        if effective_subs == 0:
            return {100: 0}
        elif effective_subs == 1:
            n_windows = 1
        elif effective_subs < 5:
            n_windows = 2
        else:
            n_windows = 3

        top_minutes = (df[(df['team_id'] == team_id) & (df['sub_in'] > self.match_initial_time)]['sub_in'].value_counts().head(n_windows).index.tolist())
        
        if len(top_minutes) < n_windows:
            if top_minutes:
                k = n_windows - len(top_minutes)
                last_minute = top_minutes[-1]
                random_minutes = np.random.randint(last_minute, 91, size=k).tolist()
                top_minutes.extend(random_minutes)
            else:
                top_minutes = np.random.randint(self.match_initial_time, 91, size=n_windows).tolist()

        base = avail_subs // n_windows
        remainder = avail_subs % n_windows
        distribution = {}
        for i in range(n_windows):
            minute = top_minutes[i]
            distribution[round(min(90, minute))] = distribution.get(minute, 0) + (base + 1 if i < remainder else base)
        return distribution

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

    def _get_teams_raw_raxg(self, minute: int, offensive_players: list, defensive_players: list, offensive_data: dict, defensive_data: dict) -> float:
        # Shared parameters
        max_fatigue_increase = 1
        improvement_rate = 0.2
        tau_inmatch = 20
        omega = 0.80
        halftime_recovery_factor = 0.85

        # Calculate updated offensive raxg (Rhythm) for each offensive player
        team_off_raxg = 0
        for player in offensive_players:
            player_data = offensive_data[player]
            initial_rhythm = player_data['initial_rhythm']
            minutes_on_field = minute - player_data.get('in')
            current_rhythm = initial_rhythm + (1 - initial_rhythm) * improvement_rate * (1 - math.exp(-minutes_on_field / tau_inmatch))

            if player_data['off_xg_coef'] >= 0:
                updated_off_raxg = player_data['off_xg_coef'] * (0.9 + 0.1 * current_rhythm)
            else:
                updated_off_raxg = player_data['off_xg_coef'] * (1 + 0.1 * current_rhythm)
            team_off_raxg += updated_off_raxg

        # Calculate updated defensive raxg (Fatigue) for each defensive player
        opp_def_raxg = 0
        for player in defensive_players:
            player_data = defensive_data[player]
            initial_fatigue = player_data['initial_fatigue']
            in_minute = player_data.get('in')

            if minute > 45 and in_minute < 45:
                minutes_played_first_half = 45 - in_minute
                fatigue_increase_first_half = (minutes_played_first_half / 45) * max_fatigue_increase * initial_fatigue
                fatigue_at_45 = initial_fatigue + fatigue_increase_first_half
                fatigue_after_halftime = fatigue_at_45 * halftime_recovery_factor
                minutes_in_second_half = minute - 45
                fatigue_increase_2nd = (minutes_in_second_half / 45) * max_fatigue_increase * fatigue_after_halftime
                current_fatigue = min(1.0, fatigue_after_halftime + fatigue_increase_2nd)
            else:
                minutes_on_field = minute - in_minute
                fatigue_increase = (minutes_on_field / 45) * max_fatigue_increase * initial_fatigue
                current_fatigue = min(1.0, initial_fatigue + fatigue_increase)

            fatigue_impact = omega * current_fatigue * abs(player_data['def_xg_coef'])

            updated_def_raxg = player_data['def_xg_coef'] - fatigue_impact 
            opp_def_raxg += updated_def_raxg

        raw_raxg = self.xg_baseline_coef + team_off_raxg - opp_def_raxg

        return raw_raxg
 
    def _get_status(self, home_goals: int, away_goals: int) -> tuple[float, float]:
        diff = home_goals - away_goals
        if diff == 0:
            return 0.0, 0.0
        elif diff == 1:
            return 0.5, -0.5
        elif diff > 1:
            return 1.5, -1.5
        elif diff == -1:
            return -0.5, 0.5
        elif diff < -1:
            return -1.5, 1.5
        
    def get_shot_type(self, rahs, rafs):
        rahs = max(0, rahs)
        rafs = max(0, rafs)
        
        total = rahs + rafs
        if total == 0:
            probs = [0.2, 0.8]
        else:
            probs = [rahs / total, rafs / total]

        selected_index = np.random.choice([0, 1], p=probs)
        return "Head" if selected_index == 0 else "Foot"
    
    def get_shooter(self, prob_dicts, body_part):
        _body_part_key = {'Head': 'headers', 'Foot': 'footers'}

        key      = _body_part_key[body_part] 
        probs    = prob_dicts['shooter'][key]
        players  = list(probs.keys())
        p_vals   = list(probs.values())
        # for player, prob in probs.items():
        #     print(f"Shooter: {player}: {prob * 100:.2f}%")
        return np.random.choice(players, p=p_vals)
    
    def get_assister(self, prob_dicts, body_part, shooter):
        _body_part_key = {'Head': 'headers', 'Foot': 'footers'}
        key      = _body_part_key[body_part]
        probs    = prob_dicts['assist'][key][shooter]
        ass      = list(probs.keys())
        p_vals   = list(probs.values())
        # for player, prob in probs.items():
        #     print(f"Assister: {player}: {prob * 100:.2f}%")
        return np.random.choice(ass, p=p_vals)

    def _insert_buf(self, rows: list, *, initial_delete: bool) -> None:
        def to_builtin(x):
            if isinstance(x, (np.generic,)):
                return x.item()
            return x
        
        if initial_delete:
            DB.execute("DELETE FROM simulation WHERE id = %s", (self.match_id,))

        batch_size = 200
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            placeholders = ', '.join(['(%s, %s, %s, %s, %s)'] * len(chunk))
            insert_sql = f"""
            INSERT INTO simulation 
                (sim_id, match_id, minute, home_goals, away_goals)
            VALUES {placeholders}
            """
            params = []
            for row in chunk:
                params.extend([to_builtin(row[0]), to_builtin(self.match_id)] + [to_builtin(x) for x in row[1:]])

            DB.execute(insert_sql, params)

    def _regulation_factors(self, ha: bool, status: float, time: int) -> float:
        ha_factor   = {True: 0.95, False: 1.05}
        status_factor = {-1.5: 1.1, -0.5: 1.05, 0: 1.0, 0.5: 0.95, 1.5: 0.9}
        time_factor = {1: 0.90, 2: 0.95, 3: 0.98, 4: 1.02, 5: 1.05, 6: 1.10}

        ha_val = ha_factor[ha]
        status_val = status_factor[status]
        time_val = time_factor[time]
        
        return ha_val * status_val * time_val

    def _get_league_regulation_data(self) -> dict:
        reg_sql = """
            SELECT
                foul_rate / 90 / 2 AS foul_rate,
                yc_rate / foul_rate AS yc_rate,
                rc_rate / foul_rate AS rc_rate
            FROM leagues
            WHERE id = %s
        """
        lrs_df = DB.select(reg_sql, (self.league_id,))
        return lrs_df.iloc[0].to_dict()

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

    def _get_team_foul_prob(self, active_players, opponent_players, status, is_home):
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

# ------------------------------ Automatization ------------------------------
class AutoLineups:
    def __init__(self, schedule_id):
        self.schedule_id = schedule_id

        print(f"Getting {get_match_title(self.schedule_id)} lineups")  

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
        match_time = result['local_time'].iloc[0]
        match_time_as_time = (datetime.min + match_time).time()
        match_timestamp = int(datetime.combine(date.today(), match_time_as_time).timestamp())

        if not match_url:
            print("No match URL provided. Skipping lineup processing.")
            return

        match = re.search(r'id:(\d+)', match_url)
        if not match:
            raise ValueError("Match ID not found in URL.")
        match_id = match.group(1)

        api_url = f"https://api.sofascore.com/api/v1/event/{match_id}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/85.0.4183.121 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        gresponse = requests.get(api_url, headers=headers)

        if gresponse.status_code != 200:
            print(f"[ERROR] API request failed with status {gresponse.status_code}. Usando Selenium como fallback.")
            event_gdata = self._fetch_json_wsel(api_url)
        else:
            event_gdata = gresponse.json()

        self.referee_name = event_gdata["event"]["referee"]["name"]

        api_lineups_url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
        lresponse = requests.get(api_lineups_url, headers=headers)

        if lresponse.status_code != 200:
            print(f"[ERROR] API request failed with status {gresponse.status_code}. Usando Selenium como fallback.")
            event_ldata = self._fetch_json_wsel(api_lineups_url)
        else:
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

        home_total_extracted = len(self.home_starters) + len(self.home_subs)
        away_total_extracted = len(self.away_starters) + len(self.away_subs)

        def _fetch_minutes(pids: list[int]) -> dict[int, int]:
            if not pids:
                return {}

            placeholders = ",".join(["%s"] * len(pids))
            sql = f"""
                SELECT player_id, minutes_played
                FROM   players_data
                WHERE  player_id IN ({placeholders})
            """
            df = DB.select(sql, tuple(pids))
            return dict(zip(df["player_id"], df["minutes_played"]))

        def _calc_strength(matched_ids: list[int], total_extracted: int) -> float:
            if total_extracted == 0:
                return 0.0

            minutes = _fetch_minutes(matched_ids)
            total_pct = 0
            for pid in matched_ids:
                total_pct += min(minutes.get(pid, 0) / 500, 1)
            return total_pct / total_extracted

        home_all_ids = home_ids_st + home_ids_bn
        away_all_ids = away_ids_st + away_ids_bn
        self.game_strength = _calc_strength(home_all_ids, home_total_extracted) * _calc_strength(away_all_ids, away_total_extracted)

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

        sql_query = """
            UPDATE schedule_data
               SET referee_name = %s,
                   game_strength = %s,
                   current_home_goals = %s,
                   current_away_goals = %s,
                   current_period_start_timestamp = %s,
                   period = %s,
                   simulate = 1
             WHERE schedule_id = %s
        """
        DB.execute(sql_query, (self.referee_name,
                               self.game_strength,
                               0,
                               0,
                               match_timestamp,
                               "period1",
                               schedule_id))

    def _fetch_json_wsel(self, url):
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

        pre_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "pre")))

        pre_content = pre_element.text
        json_data = json.loads(pre_content)

        driver.quit()
        return json_data

class AutoMatchInfo:
    def __init__(self, schedule_id):
        self.schedule_id = schedule_id

        print(f"Getting {get_match_title(self.schedule_id)} live info")    

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
        home_subs_avail = int(result['home_n_subs_avail'].iloc[0])
        away_subs_avail = int(result['away_n_subs_avail'].iloc[0])

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
            print(f"[ERROR] API request failed with status {gresponse.status_code}. Usando Selenium como fallback.")
            event_gdata = self._fetch_json_wsel(api_url)
        else:
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
            last_minute_checked: int,
            home_subs_avail: int,
            away_subs_avail: int
        ) -> tuple[list[dict], list[dict], int, int, int, int, int]:
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
                inc_type = inc.get("incidentType")

                # we only care about substitutions & cards ---------------------------
                if inc_type not in ("substitution", "card"):
                    continue

                minute = inc.get("time", 0)
                if minute <= last_minute_checked:
                    continue

                side       = "home" if inc.get("isHome") else "away"
                squad_idx  = idx_home if side == "home" else idx_away
                squad_stat = home_status if side == "home" else away_status

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
                else:  # inc_type == "card"
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

                # update “last_minute” ONLY for incidents we processed ---------------
                last_minute = max(last_minute, minute)

            # simulate flags ----------------------------------------------------------
            simulate_home = int(
                cnt["home"]["sub"] > 0 or cnt["home"]["red"] > 0 or cnt["home"]["yellow"] >= 2
            )
            simulate_away = int(
                cnt["away"]["sub"] > 0 or cnt["away"]["red"] > 0 or cnt["away"]["yellow"] >= 2
            )

            home_subs_avail = max(home_subs_avail - cnt["home"]["sub"], 0)
            away_subs_avail = max(away_subs_avail - cnt["away"]["sub"], 0)

            return (
                home_status,
                away_status,
                last_minute,
                simulate_home,
                simulate_away,
                home_subs_avail,
                away_subs_avail 
            )
        
        api_incidents_url = f"https://www.sofascore.com/api/v1/event/{match_id}/incidents"
        iresponse = requests.get(api_incidents_url, headers=headers)

        incidents_data = iresponse.json()["incidents"]

        upd_home, upd_away, last_min, sim_home, sim_away, home_subs_avail, away_subs_avail = parse_incidents(
            incidents_data,
            json.loads(home_players_data),
            json.loads(away_players_data),
            last_minute_checked,
            home_subs_avail,
            away_subs_avail
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
                period_injury_time             = %s,
                home_n_subs_avail              = %s,
                away_n_subs_avail              = %s
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
                home_subs_avail,
                away_subs_avail,
                self.schedule_id
            )
        )

    def _fetch_json_wsel(self, url):
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

        pre_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "pre")))

        pre_content = pre_element.text
        json_data = json.loads(pre_content)

        driver.quit()
        return json_data

class AutoSS:
    def __init__(self):
        active_leagues_df = DB.select("SELECT * FROM league_data WHERE is_active = 1")
        
        for league_id in tqdm(active_leagues_df["league_id"].tolist(), desc="Processing leagues"):
            league_ss_url = active_leagues_df[active_leagues_df['league_id'] == league_id]['ss_url'].values[0]

            href_list = self.get_ss_urls(league_ss_url)

            match_dict = {}

            for url in href_list:
                if "/match/" in url:
                    parts = url.split("/match/")[1].split("/")
                    if parts:
                        match = parts[0].replace("-", " ")
                        match_dict[match] = url

            missing_ssurl_games_df = DB.select(f"SELECT schedule_id, home_team_id, away_team_id FROM schedule_data WHERE league_id = {league_id} AND date >= CURRENT_DATE")

            for _, row in missing_ssurl_games_df.iterrows():
                schedule_id = int(row["schedule_id"])
                home_team = get_team_name_by_id(int(row["home_team_id"]))
                away_team = get_team_name_by_id(int(row["away_team_id"]))
                ss_url = self.get_matched_teams_url(match_dict, f"{home_team} {away_team}")

                if ss_url:
                    upd_sql = "UPDATE schedule_data SET ss_url = %s WHERE schedule_id = %s"
                    DB.execute(upd_sql, (ss_url, schedule_id))

    def get_ss_urls(self, league_ss_url):
        s = Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(league_ss_url)

        try:
            popup_close = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".Button.RVwfR")))
            popup_close.click()
        except Exception as e:
            print("Popup not found or not clickable:", e)

        driver.execute_script("window.scrollBy(0, 100);")
        round_div = driver.find_elements(By.CSS_SELECTOR, ".Box.kiSsvW")

        href_list = []
        for div in round_div:
            anchor_tags = div.find_elements(By.TAG_NAME, "a")
            for a in anchor_tags:
                href = a.get_attribute("href")
                if href:
                    href_list.append(href)

        driver.quit()
        return href_list

    def get_matched_teams_url(self, ssdict, target_title):
        def normalize_text(text):
            text = text.lower()
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            return text
        normalized_target = normalize_text(target_title)

        normalized_ssdict = {
            normalize_text(key): key for key in ssdict.keys()
        }
        
        result = process.extractOne(
            normalized_target,
            normalized_ssdict.keys(),
            scorer=fuzz.ratio,
        )

        if result is None:
            return None 
        
        match, score, _ = result

        original_match = normalized_ssdict[match]

        if score > 50:
            best_url = ssdict[original_match]
        else:
            best_url = None

        return best_url

# ------------------------------ Trading (DELETE THIS) ------------------------------
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