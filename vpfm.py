import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from fuzzywuzzy import process, fuzz
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import numpy as np
import math
import DatabaseManager

class GetPastGamesData:
    def __init__(self, league): 
        self.league = league
        self.to_date = datetime.now().date()
        # self.to_date = datetime.strptime('2025-04-23', '%Y-%m-%d').date()

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        leagues_data_query = """
        SELECT
            *  
        FROM leagues_data
        WHERE league_name = %s
        """
        self.leagues_df = self.db.select(leagues_data_query, (self.league,))

        self.from_date = self.leagues_df[self.leagues_df['league_name'] == self.league]['league_last_updated_date'].values[0]

        self.games_url, self.games_dates, self.home_teams, self.away_teams = self.get_games_info()

        for i in tqdm(range(len(self.games_url))):
            game_url = self.games_url[i]
            game_date = self.games_dates[i]
            home_team = self.home_teams[i]
            away_team = self.away_teams[i]

            try:
                self.extract_and_save_game_data(game_url, game_date, home_team, away_team)
            except Exception as e:
                print(f"Error encountered: {e}. URL: {game_url}, Date of game: {game_date}")
                raise 

        update_query = """
        UPDATE leagues_data
        SET league_last_updated_date = %s
        WHERE league_name = %s
        """
        params = (self.to_date , self.league)
        self.db.execute(update_query, params)

    def get_games_info(self):
        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        league_url = self.leagues_df[self.leagues_df['league_name'] == self.league]['league_fbref_fixtures_url'].values[0]
        table_xpath = self.leagues_df[self.leagues_df['league_name'] == self.league]['league_table_xpath'].values[0]
        driver.get(league_url)
        driver.execute_script("window.scrollTo(0, 1000);")

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, table_xpath)))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
        filtered_games_urls = []
        games_dates = []
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

            if self.from_date <= game_date < self.to_date:
                games_dates.append(game_date.strftime('%Y-%m-%d'))

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
        driver.quit()
        
        return filtered_games_urls, games_dates, home_teams, away_teams

    def extract_and_save_game_data(self, game_url, game_date, home_team, away_team):
        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(game_url)
        # driver.execute_script("window.scrollTo(0, 1000);")

        home_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="a"]/table'))) 
        home_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", home_table))

        away_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="b"]/table'))) 
        away_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", away_table))

        home_players = home_data[0][~home_data[0].iloc[:, 1].str.contains("Bench", na=False)].iloc[:, 1].tolist()
        away_players = away_data[0][~away_data[0].iloc[:, 1].str.contains("Bench", na=False)].iloc[:, 1].tolist()

        try:
            events_wrap = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="events_wrap"]')))
        except:
            return None
        
        subs_dict = {}
        max_minute = 90
        extra_first_half_minutes = 2
        extra_second_half_minutes = 5
        events = events_wrap.find_elements(By.CSS_SELECTOR, '.event')
        for event in events:
            line_list = event.text.strip().split('\n')
            for line in line_list:
                if re.match(r'^\s*45\+', line):
                    minute = line.strip("’").strip()
                    minute = int(re.search(r'\+(\d+)', minute).group(1))
                    extra_first_half_minutes = max(extra_first_half_minutes, minute)
                if re.match(r'^\s*90\+', line):
                    minute = line.strip("’").strip()
                    minute = int(re.search(r'\+(\d+)', minute).group(1))
                    extra_second_half_minutes = max(extra_second_half_minutes, minute)
            sub = event.find_elements(By.CSS_SELECTOR, '.substitute_in')
            if sub:
                for line in line_list:
                    m = re.match(r'^\s*(\d+)(?:\+\d+)?’$', line)
                    if m:
                        minute = m.group(1)
                        if int(minute) not in subs_dict:
                            subs_dict[int(minute)] = []
                    else:
                        if re.match(r'^\s*\d+\s*:\s*\d+\s*$', line):
                            continue
                        if 'for ' in line:
                            line = line.split('for ')[1].strip()                       
                        subs_dict[int(minute)].append(line)
            red_card = event.find_elements(By.CSS_SELECTOR, '.red_card')
            if red_card:
                for line in line_list:
                    if re.match(r'^\s*\d+’$', line):
                        minute = line.strip("’")
                        minute = minute.strip(" ")
                        max_minute = min(max_minute, int(minute))
        
        try:
            shots_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="shots_all"]')))
        except:
            return None

        shots_data = pd.read_html(driver.execute_script("return arguments[0].outerHTML;", shots_table))
        shots_df = shots_data[0]
        shots_df.columns = pd.MultiIndex.from_tuples(shots_df.columns)

        selected_columns = shots_df.loc[:, [
            ('Unnamed: 0_level_0', 'Minute'),
            ('Unnamed: 2_level_0', 'Squad'),
            ('Unnamed: 3_level_0', 'xG'),
            ('Unnamed: 5_level_0', 'Outcome'),
        ]]

        cleaned_columns = []
        for col in selected_columns.columns:
            if 'Unnamed' in col[0]:
                cleaned_columns.append(col[1])  
            else:
                cleaned_columns.append('_'.join(col).strip()) 

        selected_columns.columns = cleaned_columns
        selected_columns = selected_columns[selected_columns['Minute'].notna() & (selected_columns['Minute'] != 'Minute')]

        extra_minutes = selected_columns['Minute'].astype(str).str.extract(r'(\d+)\+(\d+)')

        selected_columns['Minute'] = selected_columns['Minute'].astype(str).str.replace(r'\+.*', '', regex=True).str.strip()
        selected_columns['Minute'] = selected_columns['Minute'].astype(float).astype(int)

        selected_columns['xG'] = selected_columns['xG'].astype(float)

        result = extra_minutes[extra_minutes[0].isin(['45', '90'])].groupby(0)[1].max().reindex(['45', '90'])
        extra_first_half_minutes = max(extra_first_half_minutes, int(result['45'])) if not pd.isna(result['45']) else extra_first_half_minutes
        extra_second_half_minutes = max(extra_second_half_minutes, int(result['90'])) if not pd.isna(result['90']) else extra_second_half_minutes

        home_players_data = []
        for i, player in enumerate(home_players, start=1):
            player_dict = {}
            player_dict['players_name'] = player
            player_dict['league_name'] = self.league
            player_dict['team_name'] = home_team
            player_dict['date'] = game_date

            if i <= 11:
                player_dict['sub_in'] = 0
                player_dict['sub_out'] = min(self.find_player_sub_minute(subs_dict, player) or max_minute, max_minute)
            elif i > 11:
                bench_player_played = self.find_player_sub_minute(subs_dict, player)
                if bench_player_played:
                    if bench_player_played < max_minute:
                        player_dict['sub_in'] = bench_player_played
                        player_dict['sub_out'] = max_minute
                    else:
                        continue
                else:
                    continue

            player_dict['minutes_played'] = player_dict['sub_out'] - player_dict['sub_in']
            if player_dict['sub_in'] < 45 and player_dict['sub_out'] >= 45:
                player_dict['minutes_played'] += extra_first_half_minutes
            if player_dict['sub_out'] == 90:
                player_dict['minutes_played'] += extra_second_half_minutes

            player_dict['off_xg'] = self.get_xg(player_dict['sub_in'], player_dict['sub_out'], home_team, selected_columns)
            player_dict['def_xg'] = self.get_xg(player_dict['sub_in'], player_dict['sub_out'], away_team, selected_columns)

            player_dict['in_status'] = self.get_game_status(player_dict['sub_in'], home_team, selected_columns)
            player_dict['out_status'] = self.get_game_status(player_dict['sub_out'], home_team, selected_columns)
            
            home_players_data.append(player_dict)

        away_players_data = []
        for i, player in enumerate(away_players, start=1):
            player_dict = {}
            player_dict['players_name'] = player
            player_dict['league_name'] = self.league
            player_dict['team_name'] = away_team
            player_dict['date'] = game_date

            if i <= 11:
                player_dict['sub_in'] = 0
                player_dict['sub_out'] = min(self.find_player_sub_minute(subs_dict, player) or max_minute, max_minute)
            elif i > 11:
                bench_player_played = self.find_player_sub_minute(subs_dict, player)
                if bench_player_played:
                    if bench_player_played < max_minute:
                        player_dict['sub_in'] = bench_player_played
                        player_dict['sub_out'] = max_minute
                    else:
                        continue
                else:
                    continue

            player_dict['minutes_played'] = player_dict['sub_out'] - player_dict['sub_in']
            if player_dict['sub_in'] < 45 and player_dict['sub_out'] >= 45:
                player_dict['minutes_played'] += extra_first_half_minutes
            if player_dict['sub_out'] == 90:
                player_dict['minutes_played'] += extra_second_half_minutes

            player_dict['off_xg'] = self.get_xg(player_dict['sub_in'], player_dict['sub_out'], away_team, selected_columns)
            player_dict['def_xg'] = self.get_xg(player_dict['sub_in'], player_dict['sub_out'], home_team, selected_columns)

            player_dict['in_status'] = self.get_game_status(player_dict['sub_in'], away_team, selected_columns)
            player_dict['out_status'] = self.get_game_status(player_dict['sub_out'], away_team, selected_columns)
            
            away_players_data.append(player_dict)

        columns = list(home_players_data[0].keys())
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT IGNORE INTO players_data ({columns_str}) VALUES ({placeholders})"
        for player_dict in home_players_data:
            params = tuple(player_dict[col] for col in columns)
            self.db.execute(sql, params)
        for player_dict in away_players_data:
            params = tuple(player_dict[col] for col in columns)
            self.db.execute(sql, params)

    def find_player_sub_minute(self, subs, player):
        for minute, players in subs.items():
            if player in players:
                return minute
        return None

    def get_xg(self, sub_in, sub_out, team, shots_data):
        xg = 0

        for index, row in shots_data.iterrows():
            if int(row["Minute"]) >= sub_in and int(row["Minute"]) <= sub_out:
                if fuzz.ratio(row["Squad"], team) >= 90:
                    xg += float(row["xG"])

        return xg

    def get_game_status(self, minute, team, shots_data):
        team_goals = 0
        opponent_goals = 0

        for index, row in shots_data.iterrows():
            if int(row["Minute"]) <= minute:
                if row["Outcome"] == "Goal":
                    if fuzz.ratio(row["Squad"], team) >= 90: 
                        team_goals += 1
                    elif row["Squad"] != team:
                        opponent_goals += 1

        if team_goals > opponent_goals:
            team_status = "Leading"
        elif team_goals < opponent_goals:
            team_status = "Trailing"
        elif team_goals == opponent_goals:
            team_status = "Level"

        return team_status

class GetLPGDate:
    def __init__(self, league): 
        self.league = league
        self.to_date = datetime.now().date()
        # self.to_date = datetime.strptime('2025-02-25', '%Y-%m-%d').date()

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        teams_data_query = """
        SELECT
            team_name, team_url 
        FROM teams_data
        WHERE league_name = %s
        """
        self.teams_df = self.db.select(teams_data_query, (self.league,))

        for idx, row in tqdm(self.teams_df.iterrows()):
            last_played_game_date = self.get_lpgd(row['team_url'], self.to_date)

            update_query = """
            UPDATE teams_data
            SET last_played_game = %s
            WHERE team_name = %s
            """
            params = (last_played_game_date, row['team_name'])
            self.db.execute(update_query, params)

    def get_lpgd(self, team_url, to_date):
        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(team_url)

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="matchlogs_for"]')))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
        lpgd = to_date - timedelta(days=30)

        for row in rows:
            date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
            date_text = date_element.text.strip()
            cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
            if cleaned_date_text:
                game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
            else:
                continue

            if lpgd < game_date < self.to_date:
                lpgd = game_date

        driver.quit()
        
        return lpgd

class LeagueAvg:
    def __init__(self, league):
        self.league = league

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")
        sql_query = f"""
            SELECT 
                SUM(off_xg) / NULLIF(SUM(minutes_played), 0) AS avg_xg_pm
            FROM players_data
            WHERE league_name = '{self.league}';
        """
        result = self.db.select(sql_query)

        update_query = """
        UPDATE leagues_data
        SET league_avg_xg_pm = %s
        WHERE league_name = %s
        """
        params = (result['avg_xg_pm'][0], self.league)
        self.db.execute(update_query, params)

class RemoveOldData:
    def __init__(self, league):
        self.league = league
        self.cd = datetime.now()
        self.a_year_ago_date = (self.cd - timedelta(days=365)).date()
        self.a_week_ago_date = (self.cd - timedelta(days=10)).date()

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        get_teams_query = f"""
            SELECT team_name
            WHERE league_name = {league}
            FROM teams_data;
        """

        self.teams_df = self.db.select(get_teams_query)
        self.league_teams = []
        for index, row in self.teams_df.iterrows():
            self.league_teams.append(row['team_name'])

        delete_query  = """
        DELETE FROM players_data 
        WHERE date < %s
        AND league_name = %s;
        """

        self.db.execute(delete_query, (self.a_year_ago_date, self.league))

        if self.league_teams:
            placeholders = ', '.join(['%s'] * len(self.league_teams))
            
            delete_all_query = f"""
            DELETE FROM players_data 
            WHERE league_name = %s
            AND team_name NOT IN ({placeholders})
            """
            params = (self.league,) + tuple(self.league_teams)
            
            self.db.execute(delete_all_query, params)
        else:
            print("No teams found. Skipping deletion.")

        select_deleted_data_query = """
        SELECT match_id FROM schedule_data
        WHERE match_date < %s
        """
        
        match_ids_df = self.db.select(select_deleted_data_query, (self.a_week_ago_date,))
        match_ids_list = match_ids_df["match_id"].tolist()

        if match_ids_list:
            ids_placeholders = ', '.join(['%s'] * len(match_ids_list))

            delete_schedule_query = f"""
            DELETE FROM schedule_data 
            WHERE match_id IN ({ids_placeholders})
            """
            
            self.db.execute(delete_schedule_query, tuple(match_ids_list))

            delete_sim_query = f"""
            DELETE FROM simulation_data 
            WHERE match_id IN ({ids_placeholders})
            """
            
            self.db.execute(delete_sim_query, tuple(match_ids_list))

class UpdateSchedule:
    def __init__(self, league):
        self.league = league
        self.cd = datetime.now().date()
        # self.cd = datetime.strptime('2025-02-24', '%Y-%m-%d').date()
        self.ten_days_date = self.cd + timedelta(days=10)
        self.a_month_date = self.cd - timedelta(days=30)

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")

        leagues_data_query = """
        SELECT
            *  
        FROM leagues_data
        WHERE league_name = %s
        """
        self.leagues_df = self.db.select(leagues_data_query, (self.league,))

        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        league_url = self.leagues_df[self.leagues_df['league_name'] == self.league]['league_fbref_fixtures_url'].values[0]
        table_xpath = self.leagues_df[self.leagues_df['league_name'] == self.league]['league_table_xpath'].values[0]
        driver.get(league_url)

        fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, table_xpath)))
        rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")

        for row in rows:
            date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
            date_text = date_element.text.strip()
            cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
            if cleaned_date_text:
                game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
            else:
                continue

            if self.cd <= game_date < self.ten_days_date:
                venue_time_element = row.find_element(By.CSS_SELECTOR, '.venuetime')
                venue_time_str = venue_time_element.text.strip("()")
                venue_time_obj = datetime.strptime(venue_time_str, "%H:%M").time()

                local_time_element = row.find_element(By.CSS_SELECTOR, '.localtime')
                local_time_str = local_time_element.text.strip("()")
                local_time_obj = datetime.strptime(local_time_str, "%H:%M").time()

                home_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='home_team']")
                home_name = home_name_element.text

                away_name_element = row.find_element(By.CSS_SELECTOR, "[data-stat='away_team']")
                away_name = away_name_element.text

                insert_sql = """
                INSERT INTO schedule_data
                (home_team, away_team, match_date, match_venue_time, match_local_time, league_name)
                VALUES (%s, %s, %s, %s, %s, %s);
                """

                params = (home_name, away_name, game_date, venue_time_obj, local_time_obj, self.league)
                
                self.db.execute(insert_sql, params)       

        driver.quit()
        
        delete_query  = """
        DELETE FROM schedule_data 
        WHERE match_date < %s
        AND league_name = %s
        """
        del_params = (self.a_month_date, self.league)
        
        self.db.execute(delete_query, del_params)

class AutoLineups:
    def __init__(self, league, title):
        self.league = league
        self.target_title = title

        self.db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="vpfm")
        sql_query = f"""
            SELECT 
                league_sg
            FROM leagues_data
            WHERE league_name = '{self.league}';
        """
        result = self.db.select(sql_query)
        league_url = result['league_sg'].iloc[0]

        s=Service('chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=s, options=options)
        driver.get(league_url)

        fixtures_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'content-block team-news-container')]")))
        fixtures_table = fixtures_container.find_element(By.XPATH, ".//div[contains(@class, 'fxs-table table-for-lineups')]")
        rows = fixtures_table.find_elements(By.XPATH, ".//div[contains(@class, 'table-row-loneups')]")

        for row in rows:
            fxs_game = row.find_element(By.XPATH, ".//div[contains(@class, 'fxs-game')]")
            normalized_text = " ".join(fxs_game.text.strip().split())
            score = fuzz.ratio(normalized_text, self.target_title)

            if score >= 80:
                fxs_btn = row.find_element(By.XPATH, ".//div[contains(@class, 'fxs-btn')]//a")
                driver.execute_script("arguments[0].click();", fxs_btn)

                home_lineup = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'lineups-home reverse')]")))
                away_lineup = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'lineups-away')]")))
            
                home_players_elements = home_lineup.find_elements(By.XPATH, ".//span[contains(@class, 'player-name')]")
                away_players_elements = away_lineup.find_elements(By.XPATH, ".//span[contains(@class, 'player-name')]")
                
                self.home_starters = [elem.text.strip() for elem in home_players_elements]
                self.away_starters = [elem.text.strip() for elem in away_players_elements]
                
                subs_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'lineups-teams')]")))
                teams_items = subs_container.find_elements(By.XPATH, ".//div[contains(@class, 'teams-item')]")
                
                self.home_subs = []
                self.away_subs = []
                if teams_items:
                    home_subs_elements = teams_items[0].find_elements(By.XPATH, ".//ul[contains(@class, 'lineups-sub')]/li[contains(@class, 'sub-player')]")
                    self.home_subs = [re.sub(r'^\d+\s*', '', elem.text.strip()) for elem in home_subs_elements]
                if len(teams_items) > 1:
                    away_subs_elements = teams_items[1].find_elements(By.XPATH, ".//ul[contains(@class, 'lineups-sub')]/li[contains(@class, 'sub-player')]")
                    self.away_subs = [re.sub(r'^\d+\s*', '', elem.text.strip()) for elem in away_subs_elements]

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
