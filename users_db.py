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

def add_user(telegram_id: int, stripe_customer_id: str, stripe_subscription_id: str, period_end: int):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO subscriptions (telegram_id, stripe_customer_id, stripe_subscription_id, period_end)
            VALUES (?, ?, ?, ?)
            """,
            (telegram_id, stripe_customer_id, stripe_subscription_id, period_end)
        )
        conn.commit()
        print(f"Usuario {telegram_id} agregado exitosamente.")
    except sqlite3.IntegrityError:
        print(f"El usuario con telegram_id {telegram_id} ya existe.")
    finally:
        conn.close()

yesterday = int(time.time()) - 86400
ten_more_years = int(time.time()) + 10 * 365 * 24 * 3600

# change_period_end(1138801740, ten_more_years)
