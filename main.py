# main.py

from connectors.telegram import start_telegram_bot
from connectors.api import app as fastapi_app
from memory import init_memory
from llm import manager
import threading
import uvicorn

def run_telegram():
    start_telegram_bot()

def run_api():
    # Use uvicorn to serve FastAPI app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

def main():
    manager.preload_llama_model()
    init_memory()

    t1 = threading.Thread(target=run_telegram, daemon=True)
    t2 = threading.Thread(target=run_api, daemon=True)

    t1.start()
    t2.start()

    # Keep main thread alive
    t1.join()
    t2.join()

if __name__ == "__main__":
    main()