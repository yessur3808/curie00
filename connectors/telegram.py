# connectors/telegram.py

import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from agent.core import Agent

from utils.busy import detect_busy_intent, detect_resume_intent, classify_intent_llm
from utils.persona import load_persona
from utils.session import (
    set_busy_temporarily,
    is_user_busy,
    clear_user_busy,
    small_talk_chance,
)

from memory import UserManager, ConversationManager

load_dotenv()

MASTER_USER_ID = os.getenv("MASTER_USER_ID")

# Maps telegram user_id to internal_id for this session
user_session_map = {}

# Small talk prompts for Curie
SMALL_TALK_QUESTIONS = [
    "By the way, what do you enjoy doing in your free time?",
    "Is there something new you've learned recently, mon ami?",
    "Do you have a favorite book or movie?",
    "What are you curious about these days?",
    "If you could travel anywhere, where would you go?",
    "C'est intéressant! Do you have any hobbies you love?",
]

async def handle_identify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user_id = update.message.from_user.id
    args = context.args if hasattr(context, 'args') else []
    if not args:
        await update.message.reply_text("Usage: /identify <your_secret_username>")
        return

    secret_username = args[0]
    internal_id = UserManager.get_internal_id_by_secret_username(secret_username)
    if internal_id:
        user_session_map[tg_user_id] = internal_id
        await update.message.reply_text(f"✅ Identity linked to secret_username `{secret_username}`.")
    else:
        await update.message.reply_text("❌ No user found with that secret_username.")

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greeting = context.bot_data['agent'].persona.get("greeting", "Hello!")
    await update.message.reply_text(greeting)

async def handle_busy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user_id = update.message.from_user.id
    set_busy_temporarily(tg_user_id)
    await update.message.reply_text(
        "D'accord! I'll let you focus for a while. I'll check in again later, mon ami."
    )

async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user_id = update.message.from_user.id
    clear_user_busy(tg_user_id)
    await update.message.reply_text(
        "Bienvenue! I'm here and ready to chat again. 😊"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    tg_user_id = update.message.from_user.id

    agent = context.bot_data['agent']
    telegram_username = update.message.from_user.username or f"telegram_{tg_user_id}"

    internal_id = agent.get_or_create_internal_id(
        external_id=tg_user_id,
        channel='telegram',
        secret_username=telegram_username
    )

    # --- Universal intent detection: first fast keyword, then LLM fallback ---
    message_lc = user_message.lower().strip()
    if detect_busy_intent(message_lc):
        response = agent.handle_busy(internal_id)
        await update.message.reply_text(response)
        return
    if detect_resume_intent(message_lc):
        response = agent.handle_resume(internal_id)
        await update.message.reply_text(response)
        return

    # If the message isn't caught by keywords, try LLM intent classification:
    intent = classify_intent_llm(user_message)
    if intent == "busy":
        response = agent.handle_busy(internal_id)
        await update.message.reply_text(response)
        return
    elif intent == "resume":
        response = agent.handle_resume(internal_id)
        await update.message.reply_text(response)
        return

    # --- Proceed with normal conversation ---
    agent_response = agent.handle_message(user_message, internal_id=internal_id)
    agent_response = clean_assistant_reply(agent_response)
    await update.message.reply_text(agent_response)

    # Small talk logic
    import random
    if random.random() < small_talk_chance(internal_id):
        small_talk = agent.generate_small_talk(internal_id)
        if small_talk:
            small_talk = clean_assistant_reply(small_talk)
            await update.message.reply_text(small_talk)
            

async def handle_clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_user_id = update.message.from_user.id
    telegram_username = update.message.from_user.username or f"telegram_{tg_user_id}"
    internal_id = UserManager.get_or_create_user_internal_id(
        channel='telegram',
        external_id=tg_user_id,
        secret_username=telegram_username,
        updated_by='telegram_bot'
    )

    # Only allow master user
    # Now checks by internal_id, not external Telegram ID
    from utils.db import is_master_user
    if not is_master_user(internal_id):
        await update.message.reply_text("❌ You are not authorized to use this command.")
        return
    # Check for optional argument to clear all memory
    args = context.args if hasattr(context, 'args') else []
    if args and args[0] == "all":
        ConversationManager.clear_conversation()
        await update.message.reply_text("🧹 All conversational memory cleared.")
    else:
        ConversationManager.clear_conversation(internal_id)
        await update.message.reply_text("🧹 Your conversational memory has been cleared.")

def start_telegram_bot():
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_token:
        raise RuntimeError("Telegram bot token not found in .env file or environment variables.")

    persona = load_persona()
    agent = Agent(persona=persona)

    app = ApplicationBuilder().token(telegram_token).build()
    app.bot_data['agent'] = agent

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("identify", handle_identify))
    app.add_handler(CommandHandler("busy", handle_busy))
    app.add_handler(CommandHandler("resume", handle_resume))
    app.add_handler(CommandHandler("clear_memory", handle_clear_memory))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Telegram bot is running...")
    app.run_polling()