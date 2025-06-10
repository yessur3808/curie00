# connectors/telegram.py

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from agent import Agent
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_persona():
    persona_path = os.path.join('assets', 'persona.json')
    if os.path.isfile(persona_path):
        with open(persona_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greeting = context.bot_data['agent'].persona.get("greeting", "Hello!")
    await update.message.reply_text(greeting)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = update.message.from_user.id

    # You could use session/chat context here in later phases
    agent_response = context.bot_data['agent'].handle_message(user_message, user_id=user_id)
    await update.message.reply_text(agent_response)

def start_telegram_bot():
    # Get Telegram token from environment variable
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_token:
        raise RuntimeError("Telegram bot token not found in .env file or environment variables.")

    # Load persona for this agent (optional)
    persona = load_persona()
    agent = Agent(persona=persona)

    # Create the application
    app = ApplicationBuilder().token(telegram_token).build()

    # Store the agent instance in bot_data for access in handlers
    app.bot_data['agent'] = agent

    # Add command handler for /start
    app.add_handler(CommandHandler("start", handle_start))

    # Add a message handler for text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Telegram bot is running...")
    app.run_polling()