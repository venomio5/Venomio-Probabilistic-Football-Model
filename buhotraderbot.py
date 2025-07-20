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
from telegram.error import BadRequest  
import math
import os
from dotenv import load_dotenv

load_dotenv()

locale.setlocale(locale.LC_TIME, 'C')

# stripe listen --forward-to localhost:8000/stripe-webhook

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_WEBHOOK_PORT   = 8000

BOT_USERNAME          = "BuhoTraderBot"
STRIPE_SECRET_KEY     = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PRICE_ID       = os.getenv('STRIPE_PRICE_ID')

stripe.api_key = STRIPE_SECRET_KEY

TOKEN = os.getenv('TOKEN')

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
            date,
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
        "datetime": pd.to_datetime(row["date"]),
        "home_id": int(row["home_team_id"]),
        "away_id": int(row["away_team_id"]),
        "home_goals": int(row["current_home_goals"]),
        "away_goals": int(row["current_away_goals"]),
        "period_start": row["current_period_start_timestamp"],
        "period": row["period"],
        "injury_time": int(row["period_injury_time"]) if row["period_injury_time"] and not math.isnan(row["period_injury_time"]) else 0
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
    
    max_minute = 90
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

# STRIPE
async def stripe_webhook(request):
    payload    = await request.text()
    sig_header = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except (ValueError, stripe.error.SignatureVerificationError):
            return web.Response(status=400)
    else:
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)


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
    user_df = core.DB.select("SELECT period_end FROM users_data WHERE telegram_id = %s", (user_id,))
   
    if user_df.empty:
        return False
    
    period_end = user_df.iloc[0]['period_end']
    active = bool(period_end and int(period_end) > int(time.time()))

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

    core.DB.execute(
        """
        INSERT INTO users_data (telegram_id, stripe_customer_id, stripe_subscription_id, period_end)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            stripe_customer_id     = VALUES(stripe_customer_id),
            stripe_subscription_id = VALUES(stripe_subscription_id),
            period_end             = VALUES(period_end)
        """,
        (user_id, customer_id, subscription_id, period_end)
    )

async def create_checkout_session(user_id: int) -> str:
    try:
        user_trial = core.DB.select(
            "SELECT telegram_id FROM users_data WHERE telegram_id = %s", (user_id,)
        )
        trial_data = {"trial_period_days": 7} if user_trial.empty else {}
        session = stripe.checkout.Session.create(
            mode="subscription",
            client_reference_id=str(user_id),
            subscription_data={
                "metadata": {"telegram_id": str(user_id)},
                **trial_data
            },

            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            discounts=[{"coupon": "4Noe7bX7"}],
            success_url=f"https://t.me/{BOT_USERNAME}?start=paid_{{CHECKOUT_SESSION_ID}}",
            cancel_url=f"https://t.me/{BOT_USERNAME}?start=cancel"
        )
        return session.url
    except stripe.error.InvalidRequestError as err:
        logger.error(f"Stripe price id '{STRIPE_PRICE_ID}' invalid → {err}")
        return ""

def create_billing_portal_session(user_id: int) -> str:
    user_df = core.DB.select("SELECT stripe_customer_id FROM users_data WHERE telegram_id = %s",
                (user_id,))

    if user_df.empty or not user_df.iloc[0]["stripe_customer_id"]:
        return ""
    
    customer_id = user_df.iloc[0]["stripe_customer_id"]

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"https://t.me/{BOT_USERNAME}?start=perfil",
        )
        return session.url
    except stripe.error.StripeError as err:
        logger.error(f"[Stripe] error creando portal de facturación → {err}")
        return ""

# Telegram Front End
ITEMS_PER_PAGE = 10

EVENTS_MENU = [
    ("📺Hoy", "hoy"),
    ("📅Próximos", "prox"),
    ("⬅️", "eventos"),
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
        # BotCommand("escaner",    "📈Escáner"),
        BotCommand("faq",          "❓FAQs"),
        BotCommand("perfil",    "👤Perfil"),
    ]

    await application.bot.set_my_commands(commands)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    payload = context.args[0] if context.args else ""

    if payload.startswith("paid_"):
        session_id = payload[5:]
        try:
            session = stripe.checkout.Session.retrieve(session_id)
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
        return

    description = """
🚨 Acceso exclusivo con *suscripción activa*.

🔘 *Eventos* – Obtén cuotas en tiempo real generadas por un modelo de IA avanzado que simula miles de escenarios por evento.

🕒 *Horarios* – El bot opera en horarios no oficiales por ahora, generalmente activo durante eventos deportivos. Zona horaria de referencia: GMT-6 (Monterrey, México).    

📘 Visita la sección de *Preguntas Frecuentes* y comprende a fondo cómo funciona este sistema.

⚠️ *Aviso legal*: Toda decisión que tomes es bajo tu propio criterio y responsabilidad. Este bot no garantiza resultados, solo proporciona herramientas de análisis avanzadas.
    """

    rows = [
        [InlineKeyboardButton("❓ FAQs", callback_data="faq")],
        [InlineKeyboardButton("🕊️ X", url="https://x.com/buhotrader")],
    ]

    url = await create_checkout_session(user_id)
    rows.append([InlineKeyboardButton("💳 Suscribirme", url=url)])

    markup = build_markup(rows, cols=2)

    if update.callback_query:
        await update.callback_query.edit_message_text(description, reply_markup=markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(description, reply_markup=markup, parse_mode="Markdown")

async def section_events(update: Update, context: ContextTypes.DEFAULT_TYPE):
    matches = get_all_matches()
    current_time = datetime.now()
    two_hours_ago = current_time - timedelta(hours=2.1)
    current_date = current_time.date()
    today_matches = [row for _, row in matches.iterrows() if row["date"] == current_date and ((two_hours_ago <= row["datetime"] <= current_time) or (row["datetime"] > current_time))]
    today_matches = sorted(today_matches, key=lambda row: row["datetime"])

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
            nav_buttons.append(InlineKeyboardButton("◀️", callback_data=f"today_page_{page-1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton("▶️", callback_data=f"today_page_{page+1}"))
        if nav_buttons:
            btn_rows.append(nav_buttons)

    markup = InlineKeyboardMarkup(btn_rows)

    if update.callback_query:
        await update.callback_query.edit_message_text("*Eventos de Hoy*", reply_markup=markup, parse_mode="Markdown")
    else:
        await update.message.reply_text("*Eventos de Hoy*", reply_markup=markup, parse_mode="Markdown")

async def section_scanner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "📈 *Escáner* _Funcionalidad en desarrollo_ - Recibe alertas precisas basadas en movimientos de cuotas impulsadas por el Smart Money."
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    user_df = core.DB.select(
        "SELECT stripe_subscription_id, period_end FROM users_data WHERE telegram_id = %s",
        (user_id,)
    )

    if user_df.empty:
        subscription_id, period_end = None, None
    else:
        subscription_id = user_df.iloc[0]["stripe_subscription_id"]
        period_end = user_df.iloc[0]["period_end"]

    period_end_s = (
        f"{time.strftime('%A', time.localtime(period_end)).capitalize()}, "
        f"{time.strftime('%d', time.localtime(period_end))} de "
        f"{time.strftime('%B', time.localtime(period_end)).capitalize()} del "
        f"{time.strftime('%Y', time.localtime(period_end))}"
        if period_end else "—"
    )

    auto_renew_txt = "—"
    if subscription_id:
        try:
            sub        = stripe.Subscription.retrieve(subscription_id)
            auto_renew_txt = "Sí" if not sub.get("cancel_at_period_end") else "No (cancelada)"
        except Exception as err:
            logger.warning(f"[Stripe] no se pudo recuperar la sub {subscription_id} → {err}")

    text = (
        "👤 *Tu perfil*\n\n"
        f"• Válida hasta: *{period_end_s}*\n"
        f"• Renovación automática: *{auto_renew_txt}*"
    )

    # botones (gestión / cancelación) ---------------------------------------
    buttons = []
    portal_url = create_billing_portal_session(user_id)
    if portal_url:
        buttons.append([InlineKeyboardButton("⚙️ Gestionar / cancelar plan", url=portal_url)])

    markup = InlineKeyboardMarkup(buttons) if buttons else None

    # envío / edición --------------------------------------------------------
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text, reply_markup=markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(
            text, reply_markup=markup, parse_mode="Markdown")

def build_match_header(schedule_id: int) -> str:
    match = get_all_matches().loc[lambda df: df["schedule_id"] == schedule_id].iloc[0]
    
    league  = match["league_name"]
    home    = match["home_team"]
    away    = match["away_team"]
    kickoff = match["datetime"]
    current_home_goals = int(match["current_home_goals"])
    current_away_goals = int(match["current_away_goals"])
    current_period_start_timestamp = int(match["current_period_start_timestamp"])
    period = match["period"]
    period_injury_time = int(match["period_injury_time"]) if match["period_injury_time"] and not math.isnan(match["period_injury_time"]) else 0
    game_strength = match["game_strength"]

    blocks_count = int(round(game_strength * 5))
    blocks = "■" * blocks_count + "□" * (5 - blocks_count)
    
    now = datetime.now()

    if timedelta(hours=0) <= now - kickoff <= timedelta(hours=2.1):
        if not period:
            time_display = f"⏱ Descanso  |  {current_home_goals} - {current_away_goals}"
        else:
            current_period_start = datetime.fromtimestamp(current_period_start_timestamp)
            elapsed_minutes = int((now - current_period_start).total_seconds() // 60)

            base_minute = 0
            if period == "period2":
                base_minute = 45

            current_minute = base_minute + elapsed_minutes
            max_minute = 45 if period == "period1" else 90
            capped_minute = min(current_minute, max_minute)

            if current_minute > max_minute and period_injury_time > 0:
                minute_display = f"{max_minute}+{period_injury_time}"
            else:
                minute_display = f"{capped_minute}"

            time_display = f"⏱ {minute_display}'  |  {current_home_goals} - {current_away_goals}"
    elif kickoff > now:
        time_display = "🗓 " + kickoff.strftime("%A %d de %B, %H:%M").capitalize()
    else:
        time_display = "🗓 " + kickoff.strftime("%A %d de %B, %H:%M").capitalize()

    return (
        f"<i>{league} {blocks}</i>\n"
        f"<b>{home}</b> vs <b>{away}</b>\n"
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
    rows.append([InlineKeyboardButton("⬅️", callback_data="eventos")])
    markup = build_markup(rows, cols=2)

    await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")

async def market_odds(update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str = None):
    query = update.callback_query
    await query.answer()

    data = callback_data if callback_data is not None else query.data
    _, sid, market_key = data.split("_", 2)
    schedule_id = int(sid)

    header = build_match_header(schedule_id)
    odds_text = build_market_text(schedule_id, market_key)

    text = header + odds_text

    menu = [
        [InlineKeyboardButton("⬅️", callback_data=f"match_{schedule_id}")],
        [InlineKeyboardButton("🔄", callback_data=f"reload_{schedule_id}_{market_key}")]
    ]
    markup = build_markup(menu, cols=2)

    try:
        await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        raise

def build_market_text(schedule_id: int, market_key: str) -> str:
    odds = get_odds(schedule_id, market_key)

    def fmt(val):
        return f"<code>{'N/A' if val == 10**30 else val}</code>"

    if market_key == "ft_result":
        return (
            "📊 <b>Resultado Final</b>\n"
            f"<b>Local</b>:      {fmt(odds.get('home'))}\n"
            f"<b>Visitante</b>:  {fmt(odds.get('away'))}\n"
            f"<b>Empate</b>:     {fmt(odds.get('draw'))}"
        )

    if market_key == "asian_handicap":
        lines = ["📊 <b>Handicap Asiático</b>"]
        for hcap, v in odds.items():
            inv = "-" if hcap.startswith("+") else "+"
            vis_hcap = f"{inv}{hcap[1:]}" if len(hcap) > 1 else hcap
            lines.append(f"<b>Local {hcap}</b>:      {fmt(v['home'])}")
            lines.append(f"<b>Visitante {vis_hcap}</b>: {fmt(v['away'])}")
            lines.append("")
        return "\n".join(lines).rstrip()


    if market_key == "total_goals":
        lines = ["📊 <b>Total Goles</b>"]
        for ln, p in odds.items():
            lines.append(f"<b>{ln}</b>")
            lines.append(f"⬆️ {fmt(p['over'])}")
            lines.append(f"⬇️ {fmt(p['under'])}")
            lines.append("")
        return "\n".join(lines).rstrip()


    if market_key == "correct_score":
        lines = ["📊 <b>Marcador Correcto</b>"]
        for score, odd in odds.items():
            lines.append(f"<b>{score}</b>: {fmt(odd)}")
        return "\n".join(lines)


    if market_key == "team_totals":
        lines = ["📊 <b>Total por Equipo</b>"]
        for team in ("home", "away"):
            label = "Local" if team == "home" else "Visitante"
            lines.append(f"<b>{label}</b>")
            for ln, p in odds[team].items():
                lines.append(f"  <b>{ln}</b>")
                lines.append(f"  ⬆️ {fmt(p['over'])}")
                lines.append(f"  ⬇️ {fmt(p['under'])}")
                lines.append("")
            lines.append("")
        return "\n".join(lines).rstrip()


    if market_key == "btts":
        return (
            "📊 <b>¿Ambos Anotan?</b>\n"
            f"<b>Sí</b>: {fmt(odds.get('yes'))}\n"
            f"<b>No</b>: {fmt(odds.get('no'))}"
        )

    if market_key == "player_goals":
        lines = ["📊 <b>Jugador Anota</b>"]
        for player, odd in odds.items():
            name = player.split("_")[0]
            lines.append(f"<b>{name}</b>: {fmt(odd)}")
        return "\n".join(lines)

    if market_key == "player_assists":
        lines = ["📊 <b>Jugador Asiste</b>"]
        for player, odd in odds.items():
            name = player.split("_")[0]
            lines.append(f"<b>{name}</b>: {fmt(odd)}")
        return "\n".join(lines)


    if market_key == "player_shots":
        lines = ["📊 <b>Tiros del Jugador</b>"]
        for th, plist in odds.items():
            lines.append(f"<b>+{th} tiros</b>")
            for player, odd in plist.items():
                name = player.split("_")[0]
                lines.append(f" • <b>{name}</b>: {fmt(odd)}")
            lines.append("")
        return "\n".join(lines).rstrip()

def get_odds(schedule_id: int, market_key: str) -> dict:
    shots_df, metadata = load_simulation_df(schedule_id)
    if shots_df is None or shots_df.empty or metadata.get("home_id") is None or metadata.get("away_id") is None:
        return {}

    kickoff = metadata["datetime"]
    current_period_start_timestamp = metadata["period_start"]
    period = metadata["period"]
    injury_time = metadata["injury_time"]

    now = datetime.now()

    if not (timedelta(hours=0) <= now - kickoff <= timedelta(hours=2.1)):
        current_minute = 0

    current_period_start = datetime.fromtimestamp(current_period_start_timestamp)
    elapsed_minutes = int((now - current_period_start).total_seconds() // 60)

    base_minute = 0
    if period == "period2":
        base_minute = 45

    current_minute = base_minute + min(elapsed_minutes, 45)

    if injury_time is not None:
        current_minute = current_minute - injury_time

    if period == "period1":
        extra_time = 2
        if current_minute >= 30:
            estimated = (extra_time / 15) * (current_minute - 30)
            current_minute = int(current_minute - estimated)
    elif period == "period2":
        extra_time = 5
        if current_minute >= 60:
            estimated = (extra_time / 15) * (current_minute - 60)
            current_minute = int(current_minute - estimated)

    current_minute = max(current_minute, 0)

    agg = get_aggregated_goals(shots_df, metadata.get("home_id"), current_minute, metadata.get("home_goals"), metadata.get("away_goals"))
    filtered_data = agg[
        (agg["minute"] == current_minute) &
        (agg["home_goals"] == metadata.get("home_goals")) &
        (agg["away_goals"] == metadata.get("away_goals"))
    ]
    relevant_sim_ids = filtered_data["sim_id"].unique()
    if len(relevant_sim_ids) == 0:
        return {}
    final_minute = agg["minute"].max()
    final_df = agg[(agg["minute"] == final_minute) & (agg["sim_id"].isin(relevant_sim_ids))]
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

async def reload_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, sid, market_key = query.data.split("_", 2)
    new_data = f"market_{sid}_{market_key}"
    await market_odds(update, context, callback_data=new_data)

async def section_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    questions = [
        ("📦 Suscripción y Acceso", "faq_answer_1"),
        ("⚙️ Funcionamiento del Bot", "faq_answer_2")
    ]
    markup = build_markup(questions, cols=1)
    text = "❓*FAQs*"
    if update.callback_query:
        await update.callback_query.edit_message_text(text=text, reply_markup=markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(text=text, reply_markup=markup, parse_mode="Markdown")

async def section_faq_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    faq_answers = {
        "faq_answer_1":
            """
📦 *Suscripción y Acceso*
*¿Cuánto dura la suscripción?*  
1 mes

*¿Puedo cancelar en cualquier momento?*  
Sí, desde tu perfil puedes cancelar cuando quieras. Seguirás teniendo acceso hasta el final del periodo que ya pagaste.

*¿Puedo obtener un reembolso si cancelo antes de tiempo?*  
No. Si cancelas, simplemente detienes la renovación para el próximo periodo. El mes actual seguirá activo hasta su fecha de vencimiento.

*¿Cuáles son los métodos de pago aceptados?*  
Se aceptan tarjetas de crédito y débito, además de otros métodos compatibles con Stripe como Apple Pay, Google Pay, y cuentas bancarias.
También puedes usar tarjetas recargables como Saldazo, Mercado Pago, Stori, Nu, Klar, etc., que se pueden recargar en OXXO o 7-Eleven y funcionan como débito.

*¿Puedo pausar mi suscripción temporalmente?*  
No es posible pausar la suscripción.

*¿La suscripción incluye acceso a futuras funciones?*
Sí. Solo ten en cuenta que el precio mensual puede incrementarse a medida que añadimos nuevas funciones.

*¿Puedo usarlo desde mi celular, laptop/PC o en la web?* 
Sí, puedes usarlo desde tu celular, computadora o directamente en la versión web de Telegram.

*¿La suscripción es válida para varios dispositivos o cuentas?*
La suscripción solo es válida para el número de teléfono con el que usas Telegram. No se puede compartir entre varias cuentas.

*¿Cómo afecta la zona horaria GMT-6 si estoy en otro país?*
Debes considerar la diferencia horaria respecto a GMT-6.

*¿Hay soporte técnico si tengo dudas?*
No hay soporte técnico oficial, pero si tienes dudas puedes escribirme en [X](https://x.com/BuhoTrader).
            """,
        "faq_answer_2": 
            """
⚙️ *Funcionamiento y Uso del Bot*
*¿El bot me dice en qué apostar o cómo debo usar las cuotas que ofrece?*
No se te dirá directamente en qué apostar. Lo que obtienes con la suscripción son cuotas estimadas por Buhotrader, que sirven como herramienta para complementar tu análisis.
- Si la cuota de Buhotrader es más baja que la del mercado (ya sea en casas de apuestas o intercambios), significa que Buhotrader predice que ese resultado es más probable de lo que el mercado refleja. En ese caso, se recomienda apostar a favor.
- Si la cuota de Buhotrader es más alta que la del mercado, entonces el bot estima que ese resultado es menos probable, por lo que convendría apostar en contra.

Nota: Los términos "a favor" y "en contra" son propios del trading deportivo. Si no estás familiarizado, puedes consultar un PDF explicativo que está disponible en el canal de [X](https://x.com/BuhoTrader).

*¿Voy a recibir alertas automáticas de las cuotas o tengo que estar revisando?*
No recibirás alertas automáticas de cambios en las cuotas. La recomendación es usar apps como Sofascore para configurar notificaciones de eventos clave (inicio del partido, alineaciones, goles, etc.). Con esa información, puedes revisar manualmente las cuotas en el bot en el momento más oportuno.

*¿Cómo se generan las cuotas en tiempo real?*
El modelo utiliza inteligencia artificial para simular miles de veces el desarrollo del juego con los jugadores activos. A partir de estas simulaciones, calcula las cuotas y analiza el contexto del partido en tiempo real.

*¿Cuál es la diferencia entre este bot y un tipster?*
La diferencia es que el bot es un modelo matemático. Recoge miles de datos, realiza miles de simulaciones y está entrenado con inteligencia artificial. Como resultado, ofrece probabilidades expresadas en forma de cuotas. Es más una herramienta que te permite identificar dónde puede haber valor, ya que todo depende de las cuotas del mercado.
Es imposible predecir con exactitud qué va a pasar, pero sí se puede tener una mejor idea de qué tan probable es un determinado resultado. Para obtener valor, depende de lo que el mercado ofrezca.

*¿Qué deportes/ligas cubre el bot?*
Por el momento, solo cubre fútbol y las siguientes ligas:
- Liga MX
- Campeonato Brasileiro Série A
- Major League Soccer
- Liga Profesional Argentina
- Ligue 1
- Serie A
- Premier League
- La Liga
- Bundesliga
- Primeira Liga
- Eredivisie
- Liga Belga

En el futuro se agregarán más competiciones como la NBA, la Champions League, torneos internacionales y ligas con formato de playoffs incluidos (Por el momento el modelo está entrenado únicamente para partidos de temporada regular).

*¿Cuánto dinero necesito para empezar?*
No hay un monto mínimo. Esta herramienta está pensada para apoyar tu análisis de apuestas, no para indicarte exactamente qué hacer. Tú decides cuánto arriesgar, siempre bajo tu propia responsabilidad.

*¿Hay riesgo de perder dinero?*
Sí. Como en cualquier actividad con elementos de azar, existe el riesgo de pérdida. Incluso con un buen sistema, las malas rachas son inevitables. Usa solo dinero que estés dispuesto a perder.

*¿Cómo sé que los datos son confiables?*
Cada apuesta incluye un indicador de confianza basado en el análisis del modelo. En algunos casos puede haber pocos datos disponibles (lo que reduce la fiabilidad), normalmente el modelo trabaja con datos históricos de los últimos años.

*¿Puedo ver resultados históricos del bot?*
Sí, está en desarrollo. Para evaluar el rendimiento se utilizarán métricas como el MSE (error cuadrático medio) en goles esperados, ya que simplemente contar apuestas ganadas o perdidas no refleja el verdadero rendimiento del modelo.
            """
    }
    answer_text = faq_answers.get(data, "Respuesta no encontrada.")

    rows = [("⬅️", "faq")]
    markup = build_markup(rows, cols=1)
    await query.edit_message_text(text=answer_text, reply_markup=markup, parse_mode="Markdown")

SECTIONS = {
    "start":      start,
    "eventos":     section_events,
    # "escaner":    section_scanner,
    "faq":      section_faq,
    "perfil":     section_profile,
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
        await start(update, context)
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

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    BOT_APP = app                                             

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(["eventos", "faq", "perfil"], route))
    app.add_handler(CallbackQueryHandler(match_details,  pattern=r"^match_\d+$"))
    app.add_handler(CallbackQueryHandler(market_odds,   pattern=r"^market_\d+_.+$"))
    app.add_handler(CallbackQueryHandler(reload_odds, pattern=r"^reload_\d+_.+"))
    app.add_handler(CallbackQueryHandler(section_faq, pattern=r"^faq$"))
    app.add_handler(CallbackQueryHandler(section_faq_answer, pattern=r"^faq_answer_\d+$"))

    app.add_handler(CallbackQueryHandler(section_events, pattern=r"^today_page_\d+$"))

    app.add_handler(CallbackQueryHandler(route))
    app.add_error_handler(error_handler)  

    app.run_polling()

if __name__ == "__main__":
    main()