import config
from connectors.telegram import start_telegram_bot

def main():
    # Load configuration (e.g., tokens, assistant info)
    bot_config = config.load_config()

    # Start Telegram bot, passing config
    start_telegram_bot(bot_config)

if __name__ == "__main__":
    main()