import sqlite3
import stripe
import time

def show_subs():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM subscriptions")
    rows = cur.fetchall()

    for row in rows:
        print(row)

    conn.close()

show_subs()

def change_period_end(telegram_id: int, new_period_end: int):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute(
        "UPDATE subscriptions SET period_end = ? WHERE telegram_id = ?",
        (new_period_end, telegram_id)
    )
    conn.commit()
    conn.close()

def update_customer_id(telegram_id: int, new_customer_id: str):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE subscriptions
        SET stripe_customer_id = ?
        WHERE telegram_id = ?
        """,
        (new_customer_id, telegram_id)
    )

    conn.commit()
    conn.close()

yesterday = int(time.time()) - 86400
one_more_year = int(time.time()) + 365 * 24 * 3600

change_period_end(6796842653, one_more_year)