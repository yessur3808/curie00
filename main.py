from connectors.telegram import start_telegram_bot
import memory_manager
import llm_manager


def main():
    llm_manager.preload_llama_model()
    memory_manager.init_memory()
    start_telegram_bot()

if __name__ == "__main__":
    main()