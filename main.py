from connectors.telegram import start_telegram_bot
import memory_manager


def main():
    memory_manager.init_memory()
    start_telegram_bot()

if __name__ == "__main__":
    main()