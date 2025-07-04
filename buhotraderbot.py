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

# DH
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
EVENTS_MENU = [
    ("📺En Vivo", "envivo"),
    ("📅Próximos", "prox"),
    ("🔙Regresar", "eventos"),
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

    rows.append([InlineKeyboardButton("🔙Regresar", callback_data="eventos")])

    markup = InlineKeyboardMarkup(rows)

    if update.callback_query:
        await update.callback_query.edit_message_text(
            "Eventos:", reply_markup=markup
        )
    else:
        await update.message.reply_text(
            "Eventos:", reply_markup=markup
        )

async def section_scanner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Sección Escáner"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_tutorials(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Sección Tutoriales"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Sección Perfil"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Evento en Vivo"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

async def section_upcoming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Próximos Eventos"
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

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
    "envivo":       section_live,
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
    app.add_handler(CallbackQueryHandler(route))
    app.add_error_handler(error_handler)  

    app.run_polling()

if __name__ == "__main__":
    main()