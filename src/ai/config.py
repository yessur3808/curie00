from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    MEMORY_DIR = BASE_DIR / "data" / "memory"
    MODELS_DIR = BASE_DIR / "models"
    
    CONVERSATION_HISTORY = MEMORY_DIR / "conversation_history.json"
    CUSTOM_CATEGORIES = MEMORY_DIR / "custom_categories.json"
    DEFAULT_MODEL = MODELS_DIR / "llama-2-7b-chat.gguf"
    
    MAX_SHORT_TERM_MEMORY = 10
    MAX_CATEGORY_MEMORIES = 3