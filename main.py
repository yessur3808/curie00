# main.py

from connectors.telegram import start_telegram_bot
from memory import init_memory
from llm import manager


def main():
    manager.preload_llama_model()
    init_memory()
    start_telegram_bot()

if __name__ == "__main__":
    main()