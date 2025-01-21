import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Optional

def load_environment() -> Dict[str, Optional[str]]:
    """
    Load environment variables from .env file
    
    Returns:
        Dict[str, Optional[str]]: Dictionary containing environment variables
    """
    load_dotenv()
    return {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        # Add any other environment variables you might need
        'MODEL_PATH': os.getenv('MODEL_PATH', 'models/llama-2-7b-chat.gguf'),
        'MEMORY_DIR': os.getenv('MEMORY_DIR', 'data/memory')
    }

def ensure_directory_exists(path: str) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str): Path to the directory
        
    Returns:
        Path: Path object of the created/existing directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with fallback to default value.
    
    Args:
        file_path (str): Path to the JSON file
        default (Any, optional): Default value if file doesn't exist or is invalid
        
    Returns:
        Any: Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def safe_json_save(file_path: str, data: Any) -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        file_path (str): Path to save the JSON file
        data (Any): Data to save
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        ensure_directory_exists(Path(file_path).parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")  # Consider using proper logging
        return False

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent.parent.parent

def validate_file_path(file_path: str, create_dir: bool = True) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        file_path (str): Path to validate
        create_dir (bool): Whether to create directory if it doesn't exist
        
    Returns:
        Path: Validated Path object
    """
    path = Path(file_path)
    if create_dir:
        ensure_directory_exists(path.parent)
    return path