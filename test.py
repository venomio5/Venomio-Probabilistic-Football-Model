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

class DatabaseManager:
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
        """
        Ejecuta INSERT/UPDATE/DELETE.
        ↩️  Devuelve filas afectadas.
        """
        with self._connection() as conn, self._cursor(conn) as cur:
            if many and isinstance(params, Iterable):
                cur.executemany(sql, params)  # type: ignore[arg-type]
            else:
                cur.execute(sql, params or ())
            return cur.rowcount

DB = DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")

def get_team_rest_days(team_id, target_date): 
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    team_df = DB.select(f"SELECT * FROM team_data WHERE team_id = {team_id}")

    team_fixtures_url = team_df['team_fixtures_url'].values[0]

    s=Service('chromedriver.exe')
    options = webdriver.ChromeOptions()
    #options.add_argument("--headless")
    driver = webdriver.Chrome(service=s, options=options)
    driver.get(team_fixtures_url)

    fixtures_table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="matchlogs_for"]')))
    rows = fixtures_table.find_elements(By.XPATH, "./tbody/tr")
    prev_game_date = None

    for row in rows:
        print(row)
        date_element = row.find_element(By.CSS_SELECTOR, "[data-stat='date']")
        date_text = date_element.text.strip()
        cleaned_date_text = re.sub(r'[^0-9-]', '', date_text)
        if cleaned_date_text:
            game_date = datetime.strptime(cleaned_date_text, '%Y-%m-%d').date()
        else:
            continue

        print(game_date)

        if game_date < target_date:
            if prev_game_date is None or game_date > prev_game_date:
                prev_game_date = game_date
        print(prev_game_date)

    driver.quit()

    if prev_game_date is None:
        return None

    rest_days = (target_date - prev_game_date).days
    return rest_days

print(get_team_rest_days(1, "2025-04-02"))