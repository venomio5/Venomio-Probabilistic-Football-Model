from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

import sqlite3
import stripe
import time
import logging
from aiohttp import web
import asyncio
import os
import json
from datetime import datetime, timedelta
import core
import locale
import pandas as pd

locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

# stripe listen --forward-to localhost:8000/stripe-webhook

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_WEBHOOK_PORT   = 8000

BOT_USERNAME          = "BuhoTraderBot"
STRIPE_SECRET_KEY     = "sk_test_51RgEHJQUVDoUbhrhynRb8O7wDF3G2eZGKasWW7C64ryfFYwSz7E2eVSGUdRa3u40JkgolD3vf5OuL16AMXB3cEbM00P9CoIizG"
STRIPE_PRICE_ID       = "price_1RgewoQUVDoUbhrh7K2EYz3d"

stripe.api_key = STRIPE_SECRET_KEY

TOKEN = '7731194014:AAFO4Wd_3pzrkYOMAedhQdHcCHEiwploFe8'

BOT_APP = None

logger = logging.getLogger(__name__)

# SQL
def get_all_matches():
    fixtures_query = "SELECT * FROM schedule_data"
    matches = core.DB.select(fixtures_query)

    matches["datetime"] = matches.apply(
        lambda row: datetime.combine(row["date"], datetime.min.time()) + row["local_time"],
        axis=1
    )
    matches["league_name"] = matches["league_id"].apply(lambda lid: core.get_league_name_by_id(lid))
    matches["home_team"] = matches["home_team_id"].apply(lambda tid: core.get_team_name_by_id(tid))
    matches["away_team"] = matches["away_team_id"].apply(lambda tid: core.get_team_name_by_id(tid))

    return matches

def load_simulation_df(schedule_id: int):
    shots_df = core.DB.select("SELECT * FROM simulation_data WHERE schedule_id = %s", (schedule_id,))

    meta_sql = """
        SELECT 
            home_team_id,
            away_team_id,
            current_home_goals,
            current_away_goals,
            current_period_start_timestamp,
            period,
            period_injury_time
        FROM schedule_data
        WHERE schedule_id = %s
    """

    match_df = core.DB.select(meta_sql, (schedule_id,))

    if match_df.empty:
        raise ValueError(f"schedule_id {schedule_id} not found in schedule_data.")

    row = match_df.iloc[0]

    metadata = {
        "home_id": int(row["home_team_id"]),
        "away_id": int(row["away_team_id"]),
        "home_goals": int(row["current_home_goals"]),
        "away_goals": int(row["current_away_goals"]),
        "period_start": row["current_period_start_timestamp"],
        "period": int(row["period"]),
        "injury_time": int(row["period_injury_time"])
    }

    return shots_df, metadata

def get_aggregated_goals(shots_df, home_team_id, start_minute, start_home_goals, start_away_goals):
    if shots_df is None or shots_df.empty:
        return pd.DataFrame(columns=['sim_id', 'minute', 'home_goals', 'away_goals'])
    
    df = shots_df.copy()
    df = df[df['minute'] >= start_minute]
    
    df = df.sort_values(['sim_id', 'minute']).reset_index(drop=True)
    
    df['squad']   = pd.to_numeric(df['squad'],   errors='coerce').astype('Int64')
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce').fillna(0).astype(int)
    
    df['is_home']   = df['squad'] == int(home_team_id)
    df['home_goal'] = ((df['outcome'] == 1) &  df['is_home']).astype(int)
    df['away_goal'] = ((df['outcome'] == 1) & ~df['is_home']).astype(int)
    
    df['home_goal_cum'] = df.groupby('sim_id')['home_goal'].cumsum() + start_home_goals
    df['away_goal_cum'] = df.groupby('sim_id')['away_goal'].cumsum() + start_away_goals
    
    agg = (
        df.groupby(['sim_id', 'minute'])
        .agg(home_goals=('home_goal_cum', 'last'),
            away_goals=('away_goal_cum', 'last'))
        .reset_index()
    )
    
    max_minute = int(df['minute'].max())
    full_index = pd.MultiIndex.from_product(
        [agg['sim_id'].unique(), range(max_minute + 1)],
        names=['sim_id', 'minute']
    )
    
    agg = (
        agg.set_index(['sim_id', 'minute'])
        .reindex(full_index)
        .groupby(level=0)
        .ffill()
        .fillna({'home_goals': start_home_goals, 'away_goals': start_away_goals})
        .reset_index()
    )
    
    return agg

# DB & STRIPE
def init_db():
    conn = sqlite3.connect("users.db")
    cur  = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS subscriptions(
            telegram_id             INTEGER PRIMARY KEY,
            stripe_customer_id      TEXT,
            stripe_subscription_id  TEXT,
            period_end              INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

async def stripe_webhook(request):
    payload    = await request.text()
    sig_header = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except (ValueError, stripe.error.SignatureVerificationError):
            print("[DEBUG] stripe_webhook  signature verification FAILED")
            return web.Response(status=400)
    else:
        print("[WARN] STRIPE_WEBHOOK_SECRET vacío → se omite la verificación (solo DEV)")
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)

    print("[DEBUG] stripe_webhook  event type →", event["type"])

    if event["type"] == "checkout.session.completed":
        data = event["data"]["object"]
        user_id         = int(data["client_reference_id"])
        subscription_id = data["subscription"]

        activate_subscription(user_id, subscription_id)

        if BOT_APP:
            await BOT_APP.bot.send_message(
                chat_id=user_id,
                text="✅ Suscripción activada correctamente. ¡Disfruta del servicio!"
            )

    elif event["type"] == "invoice.payment_succeeded":
        invoice = event["data"]["object"]

        subscription_id = (
            invoice.get("subscription")
            or invoice.get("subscription_details", {}).get("subscription")
            or invoice.get("lines", {}).get("data", [{}])[0].get("subscription")
        )

        if subscription_id:
            subscription = stripe.Subscription.retrieve(subscription_id)
            tg_id = subscription["metadata"].get("telegram_id")
            if tg_id:
                activate_subscription(int(tg_id), subscription_id)

    return web.Response(status=200)

async def start_stripe_webhook_server():
    app = web.Application()
    app.add_routes([web.post("/stripe-webhook", stripe_webhook)])

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(
        runner,
        host="0.0.0.0", 
        port=STRIPE_WEBHOOK_PORT
    )
    await site.start()

    print(
        f"[DEBUG] Webhook server online → http://0.0.0.0:{STRIPE_WEBHOOK_PORT}/stripe-webhook"
    )

def user_subscription_active(user_id: int) -> bool:
    conn = sqlite3.connect("users.db")
    cur  = conn.cursor()
    cur.execute("SELECT period_end FROM subscriptions WHERE telegram_id=?", (user_id,))
    row  = cur.fetchone()
    conn.close()

    active = bool(row and row[0] and row[0] > int(time.time()))

    print(f"[DEBUG] user_subscription_active({user_id}) -> {active}  row={row}") # db

    return active

def activate_subscription(user_id: int, subscription_id: str):
    sub = stripe.Subscription.retrieve(subscription_id)

    period_end = sub.get("current_period_end")
    if period_end is None: 
        try:
            period_end = sub["items"]["data"][0]["current_period_end"]
        except (KeyError, IndexError):
            period_end = 0

    customer_id = sub.get("customer")

    conn = sqlite3.connect("users.db")
    cur  = conn.cursor()
    cur.execute(
        """
        INSERT INTO subscriptions(telegram_id, stripe_customer_id,
                                   stripe_subscription_id, period_end)
        VALUES (?,?,?,?)
        ON CONFLICT(telegram_id) DO UPDATE SET
            stripe_customer_id     = excluded.stripe_customer_id,
            stripe_subscription_id = excluded.stripe_subscription_id,
            period_end             = excluded.period_end
        """,
        (user_id, customer_id, subscription_id, period_end),
    )
    conn.commit()
    conn.close()

    print(f"[DEBUG] DB updated for user {user_id}  period_end={period_end}")

async def create_checkout_session(user_id: int) -> str:
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            client_reference_id=str(user_id),


            subscription_data={
                "metadata": {"telegram_id": str(user_id)}
            },

            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"https://t.me/{BOT_USERNAME}?start=paid_{{CHECKOUT_SESSION_ID}}",
            cancel_url=f"https://t.me/{BOT_USERNAME}?start=cancel",
        )
        print(f"[DEBUG] create_checkout_session()  user={user_id}  id={session.id}  url={session.url}") # db
        return session.url
    except stripe.error.InvalidRequestError as err:
        logger.error(f"Stripe price id '{STRIPE_PRICE_ID}' invalid → {err}")
        return ""

async def ask_to_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = await create_checkout_session(update.effective_user.id)

    if not url:
        await (update.callback_query or update.message).reply_text(
            "⚠️ Ocurrió un problema al generar el enlace de pago. Intenta más tarde."
        )
        return

    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("💳 Pagar suscripción", url=url)]]
    )

    if update.callback_query:
        await update.callback_query.edit_message_text(
            "Necesitas una suscripción activa para usar el bot.",
            reply_markup=markup,
        )
    else:
        await update.message.reply_text(
            "Necesitas una suscripción activa para usar el bot.",
            reply_markup=markup,
        )

# Telegram Front End
ITEMS_PER_PAGE = 10

EVENTS_MENU = [
    ("📺Hoy", "hoy"),
    ("📅Próximos", "prox"),
    ("🔙Regresar", "eventos"),
]

MARKETS = [
    ("Resultado Final",                  "ft_result"),
    ("Handicap Asiático",                "asian_handicap"),
    ("Total de Goles (Ambos Equipos)",   "total_goals"),
    ("Marcador Correcto",                "correct_score"),
    ("Total de Goles (Por Equipo)",      "team_totals"),
    ("Jugador - Goles",                  "player_goals"),
    ("Jugador - Asistencias",            "player_assists"),
    ("Jugador - Tiros",                  "player_shots"),
    ("Ambos Equipos Anotan",             "btts")
]

def build_markup(menu, cols: int = 1):
    rows, buf = [], []

    if menu and isinstance(menu[0], list):
        flat_menu = []
        for row in menu:
            flat_menu.extend(row)
        menu = flat_menu

    for item in menu:
        if isinstance(item, InlineKeyboardButton):
            btn = item
        else:
            txt, cmd = item
            btn = InlineKeyboardButton(txt, callback_data=cmd)

        buf.append(btn)
        if len(buf) == cols:
            rows.append(buf)
            buf = []
    if buf:
        rows.append(buf)

    return InlineKeyboardMarkup(rows)

async def set_command_list(application):
    commands = [
        BotCommand("eventos",     "🔘Eventos"),
        BotCommand("escaner",    "📈Escáner"),
        BotCommand("tutoriales",  "📘Tutoriales"),
        BotCommand("perfil",    "👤Perfil"),
    ]

    await application.bot.set_my_commands(commands)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    payload = context.args[0] if context.args else ""

    print(f"[DEBUG] /start  user={user_id}  payload='{payload}'") # db

    if payload.startswith("paid_"):
        session_id = payload[5:]
        print(f"[DEBUG] Verifying Stripe session {session_id}") # db
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            print(f"[DEBUG] Stripe session retrieve → {session}") # db
            if session and session.get("subscription"):
                activate_subscription(user_id, session["subscription"])
                await update.message.reply_text(
                    "✅ Suscripción activada correctamente. ¡Disfruta del servicio!"
                )
            else:
                await update.message.reply_text(
                    "⚠️ No se encontró una suscripción válida en tu pago."
                )
        except Exception as err:
            logger.error(f"Error al validar la sesión {session_id}: {err}")
            await update.message.reply_text(
                "⚠️ Ocurrió un problema al validar tu pago. Intenta más tarde."
            )

    elif payload == "cancel":
        await update.message.reply_text("❌ El proceso de pago fue cancelado.")

    if user_subscription_active(user_id):
        print(f"[DEBUG] user {user_id} already has active subscription – skip paywall") # db
        return

    description = """
    🚨 Acceso exclusivo con suscripción activa.

    🔘 Eventos – Obtén cuotas en tiempo real generadas por un modelo de IA avanzado que simula miles de escenarios por evento.
    📈 Escáner – Recibe alertas precisas basadas en movimientos de cuotas impulsadas por el Smart Money.

    📘 Visita la sección de Preguntas Frecuentes y comprende a fondo cómo funciona este sistema.

    💼 Suscripción mensual requerida para desbloquear el acceso completo.

    ⚠️ Aviso legal: Toda decisión que tomes es bajo tu propio criterio y responsabilidad. Este bot no garantiza resultados, solo proporciona herramientas de análisis avanzadas.
    """

    rows = [
        [InlineKeyboardButton("❓ FAQs", callback_data="faq")],
        [InlineKeyboardButton("🕊️ X", url="https://x.com/buhotrader")],
    ]

    url = await create_checkout_session(user_id)
    rows.append([InlineKeyboardButton("💳 Suscribirme", url=url)])

    markup = build_markup(rows, cols=2)

    if update.callback_query:
        await update.callback_query.edit_message_text(description, reply_markup=markup)
    else:
        await update.message.reply_text(description, reply_markup=markup)

async def section_events(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = list(build_markup(EVENTS_MENU[:-1], cols=2).inline_keyboard)

    markup = InlineKeyboardMarkup(rows)

    if update.callback_query:
        await update.callback_query.edit_message_text("🔘Eventos:", reply_markup=markup)
    else:
        await update.message.reply_text("🔘Eventos:", reply_markup=markup)

async def section_scanner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Sección Escáner"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_tutorials(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text="*Texto en negrita* _cursiva_ `monoespaciado` [enlace](https://example.com)"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text, parse_mode="MarkdownV2")

async def section_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Sección Perfil"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    matches = get_all_matches()
    current_time = datetime.now()
    two_hours_ago = current_time - timedelta(hours=2.1)
    current_date = current_time.date()
    today_matches = [row for _, row in matches.iterrows() if row["date"] == current_date]

    page = 0
    if update.callback_query and update.callback_query.data.startswith("today_page_"):
        try:
            page = int(update.callback_query.data.split("_")[-1])
        except:
            page = 0

    total_pages = (len(today_matches) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    start_index = page * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    page_matches = today_matches[start_index:end_index]

    btn_rows = []
    for m in page_matches:
        button_text = f"{m['home_team']} vs {m['away_team']} ({m['datetime'].strftime('%H:%M')})"
        if two_hours_ago <= m["datetime"] <= current_time:
            button_text = "🟢 " + button_text
        btn_rows.append([InlineKeyboardButton(button_text, callback_data=f"match_{m['schedule_id']}")])

    if total_pages > 1:
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("Anterior", callback_data=f"today_page_{page-1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton("Siguiente", callback_data=f"today_page_{page+1}"))
        if nav_buttons:
            btn_rows.append(nav_buttons)

    btn_rows.append([InlineKeyboardButton("🔙Regresar", callback_data="eventos")])
    markup = InlineKeyboardMarkup(btn_rows)

    if update.callback_query:
        await update.callback_query.edit_message_text("Eventos de Hoy", reply_markup=markup)
    else:
        await update.message.reply_text("Eventos de Hoy", reply_markup=markup)

async def section_upcoming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    matches = get_all_matches()
    current_date = datetime.now().date()
    upcoming_matches = [row for _, row in matches.iterrows() if row["date"] > current_date]

    page = 0
    if update.callback_query and update.callback_query.data.startswith("upcoming_page_"):
        try:
            page = int(update.callback_query.data.split("_")[-1])
        except:
            page = 0

    total_pages = (len(upcoming_matches) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    start_index = page * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    page_matches = upcoming_matches[start_index:end_index]

    btn_rows = []
    for m in page_matches:
        button_text = f"{m['home_team']} vs {m['away_team']} ({m['datetime'].strftime('%d/%m %H:%M')})"
        btn_rows.append([InlineKeyboardButton(button_text, callback_data=f"match_{m['schedule_id']}")])

    if total_pages > 1:
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("Anterior", callback_data=f"upcoming_page_{page-1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton("Siguiente", callback_data=f"upcoming_page_{page+1}"))
        if nav_buttons:
            btn_rows.append(nav_buttons)

    btn_rows.append([InlineKeyboardButton("🔙Regresar", callback_data="eventos")])
    markup = InlineKeyboardMarkup(btn_rows)

    if update.callback_query:
        await update.callback_query.edit_message_text("📅Próximos Eventos", reply_markup=markup)
    else:
        await update.message.reply_text("📅Próximos Eventos", reply_markup=markup)

def build_match_header(schedule_id: int) -> str:
    match = get_all_matches().loc[lambda df: df["schedule_id"] == schedule_id].iloc[0]
    
    league  = match["league_name"]
    home    = match["home_team"]
    away    = match["away_team"]
    kickoff = match["datetime"]
    
    now = datetime.now()
    elapsed = now - kickoff

    if timedelta(hours=0) <= elapsed <= timedelta(hours=2.1):
        current_minute = min(int(elapsed.total_seconds() // 60), 125)
        current_score = "0 - 0"  
        time_display = f"⏱ {current_minute}'  |  {current_score}"
    elif kickoff > now:
        time_display = "🗓 " + kickoff.strftime("%A %d de %B").capitalize()
    else:
        time_display = "🗓 " + kickoff.strftime("%A %d de %B")

    return (
        f"🏆 {league}\n"
        f"{home} vs {away}\n"
        f"{time_display}\n\n"
    )

async def match_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    schedule_id = int(query.data.split("_")[1])

    text = build_match_header(schedule_id)

    rows = [
        [InlineKeyboardButton(name,
                              callback_data=f"market_{schedule_id}_{key}")]
        for name, key in MARKETS
    ]
    rows.append([InlineKeyboardButton("🔙Regresar", callback_data="eventos")])
    markup = build_markup(rows, cols=2)

    await query.edit_message_text(text, reply_markup=markup)

async def market_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, sid, market_key = query.data.split("_", 2)
    schedule_id = int(sid)

    header = build_match_header(schedule_id)
    odds_text = build_market_text(schedule_id, market_key)

    text = header + odds_text

    back_markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔙Regresar", callback_data=f"match_{schedule_id}")]]
    )

    await query.edit_message_text(text, reply_markup=back_markup)

def build_market_text(schedule_id: int, market_key: str) -> str:
    odds = get_odds(schedule_id, market_key)

    if market_key == "ft_result":
        return (
            "📊 Resultado Final\n"
            f"Local: {odds.get('home',  'N/A')}\n"
            f"Visitante: {odds.get('away', 'N/A')}\n"
            f"Empate: {odds.get('draw',  'N/A')}"
        )

    if market_key == "asian_handicap":
        return "📊 Handicap Asiático\n" + \
            "\n".join(
                f"Local {hcap}: {vals['home']}   |   Visitante {('-' if hcap.startswith('+') else '+') + hcap[1:]}: {vals['away']}"
                for hcap, vals in odds.items()
            )

    if market_key == "total_goals":
        return "📊 Total de Goles (Ambos Equipos)\n" + \
               "\n".join(
                   f"Más de {ln}: {p['over']}   |   Menos de {ln}: {p['under']}"
                   for ln, p in odds.items()
               )

    if market_key == "correct_score":
        text_lines = ["📊 Marcador Correcto"]
        for score, odd in odds.items():
            odd_str = f"{odd}" if odd != 10**30 else "N/A"
            text_lines.append(f"{score}: {odd_str}")
        return "\n".join(text_lines)
    
    if market_key == "team_totals":
        lines = ["📊 Total de Goles por Equipo"]
        for team in ["home", "away"]:
            team_name = "Local" if team == "home" else "Visitante"
            for ln, p in odds[team].items():
                lines.append(f"{team_name} Más de {ln}: {p['over']}   |   Menos de {ln}: {p['under']}")
        return "\n".join(lines)
    
    if market_key == "btts":
        return (
            "📊 Ambos Equipos Anotan\n"
            f"Sí: {odds.get('yes', 'N/A')}\n"
            f"No: {odds.get('no',  'N/A')}"
        )
    
    if market_key == "player_goals":
        lines = ["📊 Jugador - Goles"]
        for player, odd in odds.items():
            lines.append(f"{player}: {odd}")
        return "\n".join(lines)
    
    if market_key == "player_assists":
        lines = ["📊 Jugador - Asistencias"]
        for player, odd in odds.items():
            lines.append(f"{player}: {odd}")
        return "\n".join(lines)
    
    if market_key == "player_shots":
        lines = ["📊 Jugador - Tiros"]
        for threshold, players_odds in odds.items():
            lines.append(f"Más de {threshold} tiros:")
            for player, odd in players_odds.items():
                lines.append(f"{player}: {odd}")
        return "\n".join(lines)

    return "Mercado no disponible"

def get_odds(schedule_id: int, market_key: str) -> dict:
    shots_df, metadata = load_simulation_df(schedule_id)
    if shots_df is None or shots_df.empty or metadata.get("home_id") is None or metadata.get("away_id") is None:
        return {}

    agg = get_aggregated_goals(shots_df, metadata.get("home_id"), 0, metadata.get("home_goals"), metadata.get("away_goals"))
    final_minute = agg["minute"].max()
    final_df = agg[agg["minute"] == final_minute]
    total = len(final_df)
    if total == 0:
        return {}

    if market_key == "ft_result":
        home_wins = len(final_df[final_df["home_goals"] > final_df["away_goals"]])
        away_wins = len(final_df[final_df["home_goals"] < final_df["away_goals"]])
        draws     = total - home_wins - away_wins
        return {
            "home": round(1 / (home_wins / total), 3) if home_wins else 0,
            "away": round(1 / (away_wins / total), 3) if away_wins else 0,
            "draw": round(1 / (draws     / total), 3) if draws     else 0,
        }

    if market_key == "asian_handicap":
        result = {}
        for hcap in ["2.5", "1.5"]:
            val = float(hcap)
            home_plus = len(final_df[final_df["home_goals"] + val > final_df["away_goals"]])
            away_minus = total - home_plus

            home_minus = len(final_df[final_df["home_goals"] - val > final_df["away_goals"]])
            away_plus = total - home_minus

            result[f"+{hcap}"] = {
                "home": round(1 / (home_plus / total), 3) if home_plus else 0,
                "away": round(1 / (away_minus / total), 3) if away_minus else 0,
            }
            result[f"-{hcap}"] = {
                "home": round(1 / (home_minus / total), 3) if home_minus else 0,
                "away": round(1 / (away_plus / total), 3) if away_plus else 0,
            }
        return result

    if market_key == "total_goals":
        result = {}
        for t in ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"]:
            threshold = float(t)
            under = len(final_df[final_df["home_goals"] + final_df["away_goals"] < threshold])
            over  = len(final_df[final_df["home_goals"] + final_df["away_goals"] > threshold])
            result[t] = {
                "over":  round(1 / (over  / total), 3) if over  else 0,
                "under": round(1 / (under / total), 3) if under else 0,
            }
        return result

    if market_key == "correct_score":
        result = {}
        total_final_data = total
        score_counts = {}
        for _, row in final_df.iterrows():
            key = (row["home_goals"], row["away_goals"])
            score_counts[key] = score_counts.get(key, 0) + 1
        for score in [
            "0-0", "0-1", "0-2", "0-3",
            "1-0", "1-1", "1-2", "1-3",
            "2-0", "2-1", "2-2", "2-3",
            "3-0", "3-1", "3-2", "3-3"
        ]:
            home_goals, away_goals = map(int, score.split("-"))
            count = score_counts.get((home_goals, away_goals), 0)
            odds_val = round(1 / (count / total_final_data), 3) if total_final_data != 0 and count != 0 else 10**30
            result[score] = odds_val
        any_other_home_win = sum(score_counts.get((h, a), 0) for h in range(4, 11) for a in range(0, 4))
        any_other_away_win = sum(score_counts.get((h, a), 0) for h in range(0, 4) for a in range(4, 11))
        any_other_draw = sum(score_counts.get((h, h), 0) for h in range(4, 11))
        aggregated_home_odds = round(1 / (any_other_home_win / total_final_data), 3) if total_final_data != 0 and any_other_home_win != 0 else 10**30
        aggregated_away_odds = round(1 / (any_other_away_win / total_final_data), 3) if total_final_data != 0 and any_other_away_win != 0 else 10**30
        aggregated_draw_odds = round(1 / (any_other_draw / total_final_data), 3) if total_final_data != 0 and any_other_draw != 0 else 10**30
        result["Local +4"] = aggregated_home_odds
        result["Visitante +4"] = aggregated_away_odds
        result["Empate +4"] = aggregated_draw_odds
        return result
    
    if market_key == "team_totals":
        result = {"home": {}, "away": {}}
        for t in ["0.5", "1.5", "2.5"]:
            threshold = float(t)

            over_home = len(final_df[final_df["home_goals"] > threshold])
            under_home = total - over_home
            result["home"][t] = {
                "over": round(1 / (over_home / total), 3) if over_home else 0,
                "under": round(1 / (under_home / total), 3) if under_home else 0,
            }

            over_away = len(final_df[final_df["away_goals"] > threshold])
            under_away = total - over_away
            result["away"][t] = {
                "over": round(1 / (over_away / total), 3) if over_away else 0,
                "under": round(1 / (under_away / total), 3) if under_away else 0,
            }

        return result
    
    if market_key == "btts":
        both_score = len(final_df[(final_df["home_goals"] > 0) & (final_df["away_goals"] > 0)])
        not_both_score = total - both_score
        return {
            "yes": round(1 / (both_score / total), 3) if both_score else 0,
            "no":  round(1 / (not_both_score / total), 3) if not_both_score else 0,
        }

    if market_key == "player_goals":
        query = """
            SELECT shooter,
                   COUNT(DISTINCT sim_id) AS sims_with_goal,
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS total_sims,
                   COUNT(DISTINCT sim_id) * 1.0 /
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS goal_pct
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
              AND outcome = 1
            GROUP BY shooter
            ORDER BY goal_pct DESC
            LIMIT 15
        """
        params = {"schedule_id": schedule_id}
        rows = core.DB.select(query, params)
        odds = {}
        for _, row in rows.iterrows():
            pct = float(row["goal_pct"])
            odds_val = round(1 / pct, 3) if pct > 0 else 10**30
            odds[row["shooter"]] = odds_val
        return odds

    if market_key == "player_assists":
        query = """
            SELECT assister,
                   COUNT(DISTINCT sim_id) AS sims_with_assist,
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS total_sims,
                   COUNT(DISTINCT sim_id) * 1.0 /
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS assist_pct
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
              AND outcome = 1
              AND assister IS NOT NULL
            GROUP BY assister
            ORDER BY assist_pct DESC
            LIMIT 10
        """
        params = {"schedule_id": schedule_id}
        rows = core.DB.select(query, params)
        odds = {}
        for _, row in rows.iterrows():
            pct = float(row["assist_pct"])
            odds_val = round(1 / pct, 3) if pct > 0 else 10**30
            odds[row["assister"]] = odds_val
        return odds

    if market_key == "player_shots":
        thresholds = {"0.5": 1, "1.5": 2, "2.5": 3, "3.5": 4}
        total_query = """
            SELECT COUNT(DISTINCT sim_id) AS total_sims
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
        """
        total_result = core.DB.select(total_query, {"schedule_id": schedule_id})
        total_sims = total_result.iloc[0]["total_sims"] if not total_result.empty else 0
        odds = {}
        for th_str, req in thresholds.items():
            query = f"""
                SELECT shooter,
                       COUNT(DISTINCT sim_id) AS sims_with_shots
                FROM (
                    SELECT sim_id, shooter, COUNT(*) AS shot_count
                    FROM simulation_data
                    WHERE schedule_id = %(schedule_id)s
                    GROUP BY sim_id, shooter
                ) t
                WHERE shot_count >= {req}
                GROUP BY shooter
                ORDER BY sims_with_shots DESC
                LIMIT 5
            """
            params = {"schedule_id": schedule_id}
            rows = core.DB.select(query, params)
            sub_odds = {}
            for _, row in rows.iterrows():
                if total_sims > 0:
                    pct = float(row["sims_with_shots"]) / total_sims
                else:
                    pct = 0
                odds_val = round(1 / pct, 3) if pct > 0 else 10**30
                sub_odds[row["shooter"]] = odds_val
            odds[th_str] = sub_odds
        return odds

    return {}

async def section_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Preguntas frecuentes\n\n"
        "• ¿Necesito pagar para usar el bot?\n"
        "  Solo para acceder a Eventos y Escáner.\n\n"
        "• ¿Cómo me suscribo?\n"
        "  Pulsa en «Suscribirme» y sigue el proceso de pago.\n\n"
        "• ¿Puedo cancelar cuando quiera?\n"
        "  Sí, desde tu perfil en cualquier momento."
    )

    rows = [
        ("📘 Ir a tutoriales", "tutorials"),
        ("🔙Regresar", "start"),
    ]
    markup = build_markup(rows, cols=2)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await update.message.reply_text(text, reply_markup=markup)

SECTIONS = {
    "start":      start,
    "eventos":     section_events,
    "escaner":    section_scanner,
    "tutoriales": section_tutorials,
    "perfil":     section_profile,
    "hoy":       section_today,
    "prox":   section_upcoming,
    "faq":        section_faq,
}

async def route(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        action = update.message.text.lstrip("/")
    else:
        await update.callback_query.answer()
        action = update.callback_query.data

    restricted = {"eventos", "escaner", "envivo", "prox"}
    if action in restricted and not user_subscription_active(update.effective_user.id):
        await ask_to_subscribe(update, context)
        return

    handler = SECTIONS.get(action)
    if handler:
        await handler(update, context)
    else:
        msg = f"Has seleccionado «{action}»"
        if update.callback_query:
            await update.callback_query.edit_message_text(msg)
        else:
            await update.message.reply_text(msg)

async def post_init(application):
    global BOT_USERNAME
    BOT_USERNAME = (await application.bot.get_me()).username
    await set_command_list(application)

    asyncio.create_task(start_stripe_webhook_server())

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception while handling update:", exc_info=context.error)

def main():
    global BOT_APP                                             

    init_db()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    BOT_APP = app                                             

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(["eventos", "escaner", "tutoriales", "perfil"], route))
    app.add_handler(CallbackQueryHandler(match_details,  pattern=r"^match_\d+$"))
    app.add_handler(CallbackQueryHandler(market_odds,   pattern=r"^market_\d+_.+$"))
    app.add_handler(CallbackQueryHandler(route))
    app.add_error_handler(error_handler)  

    app.run_polling()

if __name__ == "__main__":
    main()