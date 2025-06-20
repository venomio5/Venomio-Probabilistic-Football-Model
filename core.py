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
import ast

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
        self.league_id = league_id
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
        options.add_argument("--headless")
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
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(team_page_url)

        nav  = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "inner_nav"))
        )
        link = nav.find_element(By.XPATH, ".//a[normalize-space(text())='Scores & Fixtures']")
        fixtures_url = link.get_attribute("href")

        driver.quit()
        return fixtures_url

DB = DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")

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

# ------------------------------ Fetch & Remove Data ------------------------------
class UpdateSchedule:
    def __init__(self, upto_date):
        self.upto_date = upto_date
        active_leagues_df = DB.select("SELECT * FROM league_data WHERE is_active = 1")
        
        for league_id in tqdm(active_leagues_df["league_id"].tolist()):
            url = active_leagues_df[active_leagues_df['league_id'] == league_id]['fbref_fixtures_url'].values[0]
            lud = active_leagues_df[active_leagues_df['league_id'] == league_id]['last_updated_date'].values[0]
            five_days_date = upto_date + timedelta(days=5)

            games_dates, games_local_time, games_venue_time, home_teams, away_teams = self.get_games_basic_info(url, lud, five_days_date)

            for i in tqdm(range(len(games_dates))):
                game_date = games_dates[i]
                game_local_time = games_local_time[i]
                game_venue_time = games_venue_time[i]
                home_team = home_teams[i]
                home_id = get_team_id_by_name(home_team)
                away_team = away_teams[i]
                away_id = get_team_id_by_name(away_team)

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
                    is_raining
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    date         = VALUES(date),
                    local_time   = VALUES(local_time),
                    venue_time   = VALUES(venue_time),
                    temperature  = VALUES(temperature),
                    is_raining   = VALUES(is_raining);
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
                    rain
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
            DB.execute(transfer_sql, (self.upto_date, league_id))

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
            DB.execute(delete_sql, (self.upto_date, league_id))

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
    
    def get_games_basic_info(self, url, lud, ten_days_date):
        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(url)
        driver.execute_script("window.scrollTo(0, 1000);")

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.stats_table")))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
        games_dates = []
        games_local_time = []
        games_venue_time = []
        home_teams = []
        away_teams = []

        for row in rows:
            date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
            date_text = date_element.text.strip()
            cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
            if cleaned_date_text:
                game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
            else:
                continue

            if lud <= game_date < ten_days_date:
                games_dates.append(game_date)

                venue_time_element = row.find_element(By.CSS_SELECTOR, '.venuetime')
                venue_time_str = venue_time_element.text.strip("()")
                venue_time_obj = datetime.strptime(venue_time_str, "%H:%M").time()
                games_venue_time.append(venue_time_obj)

                local_time_element = row.find_element(By.CSS_SELECTOR, '.localtime')
                local_time_str = local_time_element.text.strip("()")
                local_time_obj = datetime.strptime(local_time_str, "%H:%M").time()
                games_local_time.append(local_time_obj)

                home_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='home_team']")
                home_name = home_name_element.text
                home_teams.append(home_name)

                away_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='away_team']")
                away_name = away_name_element.text
                away_teams.append(away_name)
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
        options.add_argument("--headless")
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
            return None

        rest_days = (target_date - prev_game_date).days
        return rest_days

    def get_weather(self, home_id, game_date, game_venue_time):
        team_df = DB.select(f"SELECT * FROM team_data WHERE team_id = {home_id}")

        team_coordinates = team_df['team_coordinates'].values[0]
        lat, lon = team_coordinates.split(',')
        today = datetime.today().date()
        if game_date < today:
            base_url = "https://archive-api.open-meteo.com/v1/archive?"
        else:
            base_url = "https://api.open-meteo.com/v1/forecast?"

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
            raise ValueError(f"No data returned for {game_date}")

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

        avg_temp = sum(filtered_temps) / len(filtered_temps)
        rain = any(r > 0.0 for r in filtered_rains)

        return avg_temp, rain

class Extract_Data:
    def __init__(self, upto_date):
        self.upto_date = upto_date
        self.get_recent_games_match_info()
        self.update_matches_info()
        self.update_pdras()

    def get_recent_games_match_info(self):
        def get_games_basic_info(url, lud):
            s=Service('chromedriver.exe')
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
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

        def get_lineups(initial_players, sub_events, current_minute, team):
            roster_mapping = {}
            for player in initial_players:
                key = player.split("_")[0]
                roster_mapping[key] = player

            lineup = initial_players[:11]

            filtered_subs = [s for s in sub_events if s[3] == team]
            filtered_subs = sorted(filtered_subs, key=lambda x: x[0])

            for sub_minute, player_out, player_in, t in filtered_subs:
                if sub_minute > current_minute:
                    break

                for idx, player in enumerate(lineup):
                    if player.split("_")[0] == player_out:
                        if player_in in roster_mapping:
                            lineup[idx] = roster_mapping[player_in]
                        else:
                            lineup[idx] = player_in
                        break

            lineup = [roster_mapping.get(player.split("_")[0], player) for player in lineup]
            return lineup

        match_info = DB.select("SELECT match_id, url, home_team_id, away_team_id FROM match_info")
        match_detail = DB.select("SELECT match_id FROM match_detail")

        match_detail_ids = set(match_detail['match_id'].tolist())

        missing_matches_df = match_info[~match_info['match_id'].isin(match_detail_ids)]

        for _, row in tqdm(missing_matches_df.iterrows()):
            # Match detail
            s = Service('chromedriver.exe')
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            driver = webdriver.Chrome(service=s, options=options)
            driver.get(row['url'])
            home_team = get_team_name_by_id(row['home_team_id'])
            away_team = get_team_name_by_id(row['away_team_id'])
            match_id = row["match_id"]

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
                if event.find_elements(By.CSS_SELECTOR, '.goal'):
                    goal_events.append((event_minute, team))
                if event.find_elements(By.CSS_SELECTOR, '.red_card'):
                    red_events.append((event_minute, team))

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

                teamA_lineup = get_lineups(home_players, subs_events, seg_start, "home")
                teamB_lineup = get_lineups(away_players, subs_events, seg_start, "away")

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

                cum_red_home = sum(1 for minute, t in red_events if minute <= seg_end and t == "home")
                cum_red_away = sum(1 for minute, t in red_events if minute <= seg_end and t == "away")
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
            
            for key in list(home_player_stats.keys()):
                stat = home_player_stats[key]
                if stat.get("starter", True):
                    if "sub_in_min" not in stat:
                        stat["sub_in_min"] = 0
                        stat["in_status"] = "level"
                    if "sub_out_min" not in stat:
                        stat["sub_out_min"] = total_minutes
                        stat["out_status"] = "leading" if final_home_goals > final_away_goals else "level" if final_home_goals == final_away_goals else "trailing"
                    if stat["sub_out_min"] > 90:
                        stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                        stat["sub_out_min"] = 90
                    else:
                        stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                else:
                    if "sub_out_min" not in stat:
                        stat["sub_out_min"] = 0
                    if stat.get("sub_out_min") == 0:
                        del home_player_stats[key]
                    else:
                        if stat["sub_out_min"] > 90:
                            stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                            stat["sub_out_min"] = 90
                        else:
                            stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]

            for key in list(away_player_stats.keys()):
                stat = away_player_stats[key]
                if stat.get("starter", True):
                    if "sub_in_min" not in stat:
                        stat["sub_in_min"] = 0
                        stat["in_status"] = "level"
                    if "sub_out_min" not in stat:
                        stat["sub_out_min"] = total_minutes
                        stat["out_status"] = "leading" if final_away_goals > final_home_goals else "level" if final_away_goals == final_home_goals else "trailing"
                    if stat["sub_out_min"] > 90:
                        stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                        stat["sub_out_min"] = 90
                    else:
                        stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                else:
                    if "sub_out_min" not in stat:
                        stat["sub_out_min"] = 0
                    if stat.get("sub_out_min") == 0:
                        del away_player_stats[key]
                    else:
                        if stat["sub_out_min"] > 90:
                            stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]
                            stat["sub_out_min"] = 90
                        else:
                            stat["minutes_played"] = stat["sub_out_min"] - stat["sub_in_min"]

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

            for idx, shot in all_shots.iterrows():
                shooter_name = shot["Player"].strip()
                shot_body = shot["Body Part"]
                shot_xg = float(shot["xG"])
                shot_psxg = float(shot["PSxG"])
                outcome = shot["Outcome"]
                sca_event = shot["SCA 1_Event"]
                sca_player = shot["SCA 1_Player"].strip()

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
                shot_minute = int(shot["Minute"])

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

                match_segment = str(min((shot_minute // 15) + 1, 6))

                cum_red_home = sum(1 for minute, t in red_events if minute <= shot_minute and t == "home")
                cum_red_away = sum(1 for minute, t in red_events if minute <= shot_minute and t == "away")

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

                sql_shot = "INSERT IGNORE INTO shots_data (match_id, xg, psxg, shooter_id, assister_id, GK_id, off_players, def_players, match_state, match_segment, player_dif, shot_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                params_shot = (match_id,
                               shot_xg,
                               shot_psxg,
                               shooter_id,
                               assister_id,
                               GK_id,
                               json.dumps(off_players, ensure_ascii=False),
                               json.dumps(def_players, ensure_ascii=False),
                               match_state,
                               match_segment,
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
        non_pdras_matches_df = DB.select("SELECT detail_id, match_id, teamA_players, teamB_players, minutes_played FROM match_detail WHERE teamA_pdras IS NULL OR teamB_pdras IS NULL;")

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

        for _, row in non_pdras_matches_df.iterrows():
            minutes = row['minutes_played']
            teamA_ids = row['teamA_players']
            teamB_ids = row['teamB_players']

            teamA_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_pdras = (teamA_offense - teamB_defense) * minutes

            teamB_offense = sum(off_sh_coef_dict.get(p, 0) for p in teamB_ids)
            teamA_defense = sum(def_sh_coef_dict.get(p, 0) for p in teamA_ids)
            teamB_pdras = (teamB_offense - teamA_defense) * minutes

            DB.execute(
                "UPDATE match_detail SET teamA_pdras = %s, teamB_pdras = %s WHERE detail_id = %s",
                (teamA_pdras, teamB_pdras, row['detail_id'])
            )

# ------------------------------ Process data ------------------------------
class Process_Data:
    def __init__(self):
        """
        Class to reset the players_data table and fill it with new data.
        """

        DB.execute("TRUNCATE TABLE players_data;")

        self.insert_players_basics()
        self.update_players_shots_coef()
        self.update_players_totals()
        self.update_players_xg_coef()

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
            for shot_type in ["headers", "footers"]:
                league_matches_df = DB.select(f"SELECT match_id FROM match_info WHERE league_id = {league_id}")
                matches_ids = league_matches_df['match_id'].tolist()
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

                X = sp.csr_matrix((data_vals, (rows, cols)), shape=(row_num, 2 * num_players))
                y_array = np.array(y)
                sample_weights_array = np.array(sample_weights)

                ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg')
                ridge.fit(X, y_array, sample_weight=sample_weights_array)

                offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
                defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))

                for player in players:
                    off_sh = offensive_ratings[player]
                    def_sh = defensive_ratings[player]
                    update_coef_query = f"""
                    UPDATE players_data
                    SET off_{shot_type}_coef = %s, def_{shot_type}_coef = %s
                    WHERE player_id = %s
                    """
                    DB.execute(update_coef_query, (off_sh, def_sh, player))
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
                if status_row["sub_in"]:
                    subs_in_list.append(status_row["sub_in"])
                if status_row["sub_out"]:
                    subs_out_list.append(status_row["sub_out"])

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
                json.dumps(subs_in_list),
                json.dumps(subs_out_list),
                player_id
            ))

        # referee
        referee_df = DB.select("SELECT DISTINCT referee_name FROM match_info")
        
        for referee in referee_df["referee_name"].tolist():
            ragg_query = """
            SELECT
                COALESCE(SUM(total_fouls), 0) AS fouls,
                COALESCE(SUM(yellow_cards), 0) AS yellow_cards,
                COALESCE(SUM(red_cards), 0) AS red_cards,
                COUNT(*) * 90 AS minutes_played
            FROM match_info
            WHERE referee_name = %s
            """
            ragg_result = DB.select(ragg_query, (referee,))
            if ragg_result.empty:
                continue

            row = ragg_result.iloc[0]

            rupdate_query = """
            UPDATE referee_data
            SET 
                fouls = %s,
                yellow_cards = %s,
                red_cards = %s,
                minutes_played = %s 
            WHERE referee_name = %s
            """
            DB.execute(rupdate_query, (
                int(row["fouls"]),
                int(row["yellow_cards"]),
                int(row["red_cards"]),
                int(row["minutes_played"]),
                referee
            ))

    def update_players_xg_coef(self):
        """
        Function to update players xg coefficients per league
        """
        league_id_df = DB.select("SELECT league_id FROM league_data")
        
        for league_id in league_id_df['league_id'].tolist():
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
                
                ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg')
                ridge.fit(X, y_array, sample_weight=sample_weights_array)
                
                offensive_ratings = dict(zip(players, ridge.coef_[:num_players]))
                defensive_ratings = dict(zip(players, ridge.coef_[num_players:]))
                
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

# ------------------------------ Monte Carlo ------------------------------
class Alg:
    def __init__(self, home_team, away_team, home_lineups, away_lineups, league, match_date, match_time, match_id, home_initial_goals, away_initial_goals, match_initial_time, home_n_subs, away_n_subs, home_n_rc, away_n_rc):
        self.home_team = home_team
        self.away_team = away_team
        self.league = league
        self.match_date = match_date
        self.match_time = match_time
        self.match_id = match_id
        self.home_lineups = home_lineups
        self.away_lineups = away_lineups
        self.home_initial_goals = home_initial_goals
        self.away_initial_goals = away_initial_goals
        self.match_initial_time = match_initial_time
        self.home_n_subs = home_n_subs
        self.away_n_subs = away_n_subs
        self.home_n_rc = home_n_rc
        self.away_n_rc = away_n_rc

        self.home_starters, self.home_subs = self.divide_matched_players(self.home_team, self.league, self.home_lineups)
        self.away_starters, self.away_subs = self.divide_matched_players(self.away_team, self.league, self.away_lineups)

        self.home_players_data = self.get_players_data(self.home_team, self.home_starters, self.home_subs, self.league)
        self.away_players_data = self.get_players_data(self.away_team, self.away_starters, self.away_subs, self.league)

        self.league_avg = self.get_league_avg(self.league)
        self.home_sub_minutes, self.away_sub_minutes = self.get_sub_minutes(self.home_team, self.away_team, self.league, self.match_initial_time, self.home_n_subs, self.away_n_subs)
        self.all_sub_minutes = list(set(list(self.home_sub_minutes.keys()) + list(self.away_sub_minutes.keys())))

        self.home_features = self.compute_features(self.home_team, self.away_team, self.league, True, self.match_date, self.match_time)
        self.away_features = self.compute_features(self.away_team, self.home_team, self.league, False, self.match_date, self.match_time)

        if self.match_initial_time >= 75:
            range_value = 10000
        elif self.match_initial_time >= 60:
            range_value = 20000
        elif self.match_initial_time >= 45:
            range_value = 30000
        elif self.match_initial_time >= 30:
            range_value = 40000
        elif self.match_initial_time >= 15:
            range_value = 50000
        elif self.match_initial_time < 15:
            range_value = 60000

        all_rows = []

        for i in tqdm(range(range_value)):
            home_goals = self.home_initial_goals
            away_goals = self.away_initial_goals
            home_active_players = self.home_starters
            away_active_players = self.away_starters
            home_passive_players = self.home_subs
            away_passive_players = self.away_subs
            sql_momentum = 0

            home_last_goal_time = None
            away_last_goal_time = None

            home_natural_projected_goals = self.get_teams_natural_projected_goals(home_active_players, away_active_players, self.home_players_data, self.away_players_data, self.league_avg)
            away_natural_projected_goals = self.get_teams_natural_projected_goals(away_active_players, home_active_players, self.away_players_data, self.home_players_data, self.league_avg)

            self.home_rc_offensive_penalty = 1 if self.home_n_rc == 0 else 0.5 ** self.home_n_rc
            self.home_rc_defensive_penalty = 1 if self.home_n_rc == 0 else 1.6 ** self.home_n_rc

            self.away_rc_offensive_penalty = 1 if self.away_n_rc == 0 else 0.5 ** self.away_n_rc
            self.away_rc_defensive_penalty = 1 if self.away_n_rc == 0 else 1.6 ** self.away_n_rc

            for minute in range(self.match_initial_time, 91):
                all_rows.append((i, minute, home_goals, away_goals, sql_momentum))
                home_status, away_status = self.get_status(home_goals, away_goals)
                time_segment = self.get_time_segment(minute)

                if minute in self.all_sub_minutes:
                    if minute in list(self.home_sub_minutes.keys()):
                        home_active_players, home_passive_players = self.swap_players(home_active_players, home_passive_players, self.home_players_data, self.home_sub_minutes[minute], home_status)
                    if minute in list(self.away_sub_minutes.keys()):
                        away_active_players, away_passive_players = self.swap_players(away_active_players, away_passive_players, self.away_players_data, self.away_sub_minutes[minute], away_status)
                    home_natural_projected_goals = self.get_teams_natural_projected_goals(home_active_players, away_active_players, self.home_players_data, self.away_players_data, self.league_avg)
                    away_natural_projected_goals = self.get_teams_natural_projected_goals(away_active_players, home_active_players, self.away_players_data, self.home_players_data, self.league_avg)

                home_recent = home_last_goal_time is not None and (minute - home_last_goal_time) <= 10
                away_recent = away_last_goal_time is not None and (minute - away_last_goal_time) <= 10

                if home_last_goal_time is None and away_last_goal_time is None:
                    last_team = None
                elif away_last_goal_time is None or (home_last_goal_time and home_last_goal_time > away_last_goal_time):
                    last_team = "home"
                else:
                    last_team = "away"

                home_momentum = 1.02 if (home_recent and last_team == "home") else 1.0
                away_momentum = 1.02 if (away_recent and last_team == "away") else 1.0

                sql_momentum = 1 if home_momentum > 1.0 else 2 if away_momentum > 1.0 else 0

                home_proj_xg = home_natural_projected_goals * self.home_features['hna'] * self.home_features[time_segment] * self.home_features[home_status] * self.home_features['Travel'] * self.home_features['Elevation'] * self.home_features['Rest'] * self.home_features['Time'] * home_momentum * self.home_rc_offensive_penalty * self.away_rc_defensive_penalty
                away_proj_xg = away_natural_projected_goals * self.away_features['hna'] * self.away_features[time_segment] * self.away_features[away_status] * self.away_features['Travel'] * self.away_features['Elevation'] * self.away_features['Rest'] * self.away_features['Time'] * away_momentum * self.home_rc_defensive_penalty * self.away_rc_offensive_penalty

                home_goals_scored = np.random.poisson(home_proj_xg)
                away_goals_scored = np.random.poisson(away_proj_xg)

                home_goals += home_goals_scored
                away_goals += away_goals_scored

        self.insert_sim_data(all_rows, self.match_id)

    def divide_matched_players(self, team, league, lineups):
        clean_list = [line for line in lineups.split('\n') if line and not any(char.isdigit() for char in line)]

        unmatched_starters = clean_list[:11]
        unmatched_subs = clean_list[11:]

        db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")
        sql_query = """
            SELECT DISTINCT players_name
            FROM players_data
            WHERE league_name = %s
            AND team_name = %s;
        """
        players_df = db.select(sql_query, (league, team))
        db_players = [row['players_name'] for idx, row in players_df.iterrows()]

        matched_starters = []
        matched_subs = []

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
        while unmatched_subs:
            remaining_players = []
            for player in unmatched_subs:
                closest_match = process.extractOne(player, db_players, score_cutoff=threshold)
                if closest_match:
                    matched_subs.append(closest_match[0])
                    db_players.remove(closest_match[0])
                else:
                    remaining_players.append(player)
            if not remaining_players:
                break
            unmatched_subs = remaining_players
            threshold -= 20

        return matched_starters, matched_subs

    def get_players_data(self, team, team_starters, team_subs, league):
        all_players = team_starters + team_subs
        players_dict = {}

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")
        escaped_players = [player.replace("'", "''") for player in all_players]

        team_starters_str = ", ".join([f"'{player}'" for player in escaped_players])

        sql_query = f"""
            SELECT 
                players_name, off_xg, def_xg, minutes_played, sub_in, sub_out, in_status, out_status
            FROM players_data
            WHERE league_name = '{league}'
            AND team_name = '{team}'
            AND players_name IN ({team_starters_str});
        """
        result = self.db.select(sql_query)

        for player in all_players:
            in_status_dict = {'Leading': 0, 'Level': 0, 'Trailing': 0}
            out_status_dict = {'Leading': 0, 'Level': 0, 'Trailing': 0}
            player_off_xg = float(sum(result[result['players_name'] == player]['off_xg']))
            player_def_xg = float(sum(result[result['players_name'] == player]['def_xg']))
            player_minutes = int(sum(result[result['players_name'] == player]['minutes_played']))

            in_status_data = (
                result[(result['players_name'] == player) & (result['sub_in'] != 0)]
                .groupby('in_status')
                .size()
            )

            if not in_status_data.empty:
                in_status_dict.update(in_status_data.to_dict())

                total_in = sum(in_status_dict.values())
                for key in in_status_dict:
                    in_status_dict[key] = (in_status_dict[key] / total_in)

            out_status_data = (
                result[(result['players_name'] == player) & (result['sub_out'] <= 90)]
                .groupby('out_status')
                .size()
            )

            if not out_status_data.empty:
                out_status_dict.update(out_status_data.to_dict())

                total_out = sum(out_status_dict.values())
                for key in out_status_dict:
                    out_status_dict[key] = (out_status_dict[key] / total_out)

            players_dict[player] = {'off_xg': player_off_xg, 'def_xg': player_def_xg, 'total_minutes': player_minutes, 'in_status_dict': in_status_dict, 'out_status_dict': out_status_dict}

        return players_dict 

    def get_league_avg(self, league):
        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        league_avg_query = f"""
            SELECT 
                league_avg_xg_pm
            FROM leagues_data
            WHERE league_name = '{league}';
        """

        league_result = self.db.select(league_avg_query)

        return float(league_result['league_avg_xg_pm'][0])  

    def get_sub_minutes(self, team, opponent, league, match_initial_time, home_n_subs_avail, away_n_subs_avail):
        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        teams_data_query = f"""
            SELECT 
                team_name, date, sub_in, sub_out, in_status, out_status
            FROM players_data
            WHERE league_name = '{league}'
            AND team_name IN ('{team}', '{opponent}');
        """

        query_df = self.db.select(teams_data_query)

        team_avg_subs = round(
            query_df[(query_df['team_name'] == team) & (query_df['sub_in'] != 0)]
            .groupby('date')
            .size()
            .mean()
        )

        opponent_avg_subs = round(
            query_df[(query_df['team_name'] == opponent) & (query_df['sub_in'] != 0)]
            .groupby('date')
            .size()
            .mean()
        )

        effective_team_subs = max(0, min(team_avg_subs - (5 - home_n_subs_avail), home_n_subs_avail))
        effective_opponent_subs = max(0, min(opponent_avg_subs - (5 - away_n_subs_avail), away_n_subs_avail))

        def get_distribution(team_name, avail_subs):
            if avail_subs == 0:
                return {100: 0}
            if avail_subs == 1:
                n_windows = 1
            elif avail_subs < 5:
                n_windows = 2
            else:
                n_windows = 3

            top_minutes = (
                query_df[(query_df['team_name'] == team_name) &
                        (query_df['sub_in'] != 0) &
                        (query_df['sub_in'] > match_initial_time)]
                ['sub_in']
                .value_counts()
                .head(n_windows)
                .index
                .tolist()
            )

            base = avail_subs // n_windows
            remainder = avail_subs % n_windows
            distribution = {}
            for i in range(n_windows):
                distribution[top_minutes[i]] = base + 1 if i < remainder else base
            return distribution

        team_distribution = get_distribution(team, effective_team_subs)
        opponent_distribution = get_distribution(opponent, effective_opponent_subs)

        return team_distribution, opponent_distribution

    def swap_players(self, active_players, passive_players, players_data, subs, game_status):
        total_active_minutes = 0
        for player in active_players:
            total_active_minutes += players_data[player]['total_minutes']

        active_players_dict = {}

        for player in active_players:
            active_players_dict[player] = (players_data[player]['total_minutes'] / total_active_minutes) * (players_data[player]['out_status_dict'][game_status])

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
            total_passive_minutes += players_data[player]['total_minutes']

        passive_players_dict = {}

        for player in passive_players:
            passive_players_dict[player] = (players_data[player]['total_minutes'] / total_passive_minutes) * (players_data[player]['in_status_dict'][game_status])

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

    def get_teams_natural_projected_goals(self, offensive_players, defensive_players, offensive_data, defensive_data, league_avg):
        total_off_xg = 0
        total_off_minutes_played = 0
        for player in offensive_players:
            total_off_xg += offensive_data[player]['off_xg']
            total_off_minutes_played += offensive_data[player]['total_minutes']
        total_off_pm = total_off_xg/total_off_minutes_played

        total_def_xg = 0
        total_def_minutes_played = 0
        for player in defensive_players:
            total_def_xg += defensive_data[player]['def_xg']
            total_def_minutes_played += defensive_data[player]['total_minutes']
        total_def_pm = total_def_xg/total_def_minutes_played

        team_natural_proj_goals = (total_off_pm/league_avg)*(total_def_pm/league_avg)*league_avg

        return team_natural_proj_goals

    def compute_features(self, team, opponent, league, is_home, match_date, match_time):
        if is_home:
            hna_mp = 1.1
            distance_mp = self.compute_travel_distance_mp(team, opponent, league)
            elevation_mp = self.compute_elevation_mp(opponent, team, league)
        else:
            hna_mp = 0.9
            distance_mp = 1
            elevation_mp = 1

        rest_mp = self.compute_rest_mp(opponent, league, match_date)
        time_mp = self.compute_time_mp(match_time)

        features = {
            'hna': hna_mp,
            '00-15': 0.85,
            '15-30': 0.95,
            '30-45': 1.05,
            '45-60': 1.03,
            '60-75': 1.02,
            '75-90': 1.1,
            'Level': 0.97,
            'Trailing': 1.02,
            'Leading': 1.01,
            'Travel': distance_mp,
            'Elevation': elevation_mp,
            'Rest': rest_mp,
            'Time': time_mp
        }

        return features
    
    def compute_travel_distance_mp(self, team, opponent, league):
        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        location_query = f"""
            SELECT 
                team_name, coordinates
            FROM teams_data
            WHERE league_name = '{league}'
            AND team_name IN ('{team}', '{opponent}');
        """
        location_df = self.db.select(location_query)

        team_coordinates = location_df[location_df['team_name'] == team]['coordinates'].values[0]
        lat1, lon1 = map(str.strip, team_coordinates.split(','))
        opponent_coordinates = location_df[location_df['team_name'] == opponent]['coordinates'].values[0]
        lat2, lon2 = map(str.strip, opponent_coordinates.split(','))
    
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
        
        increments = distance // 100
        
        penalization_percent = increments * 1
        
        if penalization_percent > 20:
            penalization_percent = 20
        
        result = 1 + (penalization_percent / 100)
        
        return result
    
    def compute_elevation_mp(self, team, opponent, league):
        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        elevation_query = f"""
            SELECT 
                team_name, elevation
            FROM teams_data
            WHERE league_name = '{league}'
            AND team_name IN ('{team}', '{opponent}');
        """
        elevation_df = self.db.select(elevation_query)

        team_elevation = elevation_df[elevation_df['team_name'] == team]['elevation'].values[0]
        opponent_elevation = elevation_df[elevation_df['team_name'] == opponent]['elevation'].values[0]

        elevation_dif = opponent_elevation - team_elevation
        
        increments = elevation_dif // 100
        penalization_percent = 0

        if increments > 0:
            penalization_percent = min(increments * 1, 15)
            result = 1 + (penalization_percent / 100)

        elif increments < 0:
            penalization_percent = max(increments * 1, -15)
            result = 1 + (penalization_percent / 100)
        
        result = 1 + (penalization_percent / 100)
        
        return result
    
    def compute_rest_mp(self, opponent, league, match_date):
        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        lpg_query = f"""
            SELECT 
                team_name, last_played_game
            FROM teams_data
            WHERE league_name = '{league}'
            AND team_name = '{opponent}';
        """
        lpg_df = self.db.select(lpg_query)

        opponent_lpg = lpg_df[lpg_df['team_name'] == opponent]['last_played_game'].values[0]

        opponent_rest_days = (match_date - opponent_lpg).days

        rest_days = {2: 15, 7: -3, 14: 10}
        sorted_items = sorted(rest_days.items())
        
        if opponent_rest_days <= sorted_items[0][0]:
            rest_opponent_adjustment = sorted_items[0][1]
        elif opponent_rest_days >= sorted_items[-1][0]:
            rest_opponent_adjustment = sorted_items[-1][1]
        else:
            for (x0, y0), (x1, y1) in zip(sorted_items[:-1], sorted_items[1:]):
                if x0 <= opponent_rest_days <= x1:
                    rest_opponent_adjustment = y0 + (opponent_rest_days - x0) * (y1 - y0) / (x1 - x0)
                    break
        penalization_percent = min(rest_opponent_adjustment, 15)
        result = 1 + (penalization_percent / 100)
        
        return result
    
    def compute_time_mp(self, match_time):
        total_seconds = match_time.total_seconds()
        hours = int(total_seconds // 3600) % 24

        if 0 <= hours < 13:
            category = "Afternoon"
        elif 13 <= hours < 20:
            category = "Evening"
        elif 20 <= hours <= 23:
            category = "Night"

        if category == "Afternoon":
            penalization_percent = 6
        elif category == "Evening":
            penalization_percent = -2
        elif category == "Night":
            penalization_percent = -4
        
        result = 1 - (penalization_percent / 100)
        
        return result
    
    def get_status(self, home_goals, away_goals):
        if home_goals > away_goals:
            return "Leading", "Trailing"
        elif home_goals < away_goals:
            return "Trailing", "Leading"
        else:
            return "Level", "Level"

    def get_time_segment(self, minute):
        if minute < 15:
            return "00-15"
        elif minute < 30:
            return "15-30"
        elif minute < 45:
            return "30-45"
        elif minute < 60:
            return "45-60"
        elif minute < 75:
            return "60-75"
        else:
            return "75-90"
        
    def insert_sim_data(self, rows, match_id):
        delete_query  = """
        DELETE FROM simulation_data 
        WHERE match_id = %s
        """

        self.db.execute(delete_query, (match_id,))

        batch_size = 200
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            placeholders = ', '.join(['(%s, %s, %s, %s, %s, %s)'] * len(chunk))
            insert_sql = f"""
            INSERT INTO simulation_data 
                (match_id, sim_id, minute, home_goals, away_goals, momentum)
            VALUES {placeholders}
            """
            params = []
            for row in chunk:
                params.extend([match_id] + list(row))

            self.db.execute(insert_sql, params)    

# ------------------------------ Trading ------------------------------