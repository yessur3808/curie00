# src/ai/memory/categories.py:
import json
import os
from pathlib import Path
from typing import Dict
from src.utils.helpers import ensure_directory_exists


class DynamicMemoryCategory:
    def __init__(self):
        # Default categories
        self._categories = {
            "PERSONAL": "User preferences, names, personal details",
            "TECHNICAL": "Technical discussions and explanations",
            "EMOTIONAL": "Emotional responses and sentiments",
            "FACTUAL": "General facts and information",
            "CONTEXTUAL": "Conversation context and flow",
            "SCIENTIFIC": "Scientific discussions and data",
            "PREFERENCE": "User preferences and settings"
        }
        self._load_custom_categories()

    def _load_custom_categories(self):
        try:
            with open("memory/custom_categories.json", "r") as f:
                custom_categories = json.load(f)
                self._categories.update(custom_categories)
        except FileNotFoundError:
            pass

    def _save_custom_categories(self):
        os.makedirs("memory", exist_ok=True)
        with open("memory/custom_categories.json", "w") as f:
            json.dump(self._categories, f, indent=2)

    def add_category(self, name: str, description: str) -> bool:
        """Dynamically add a new category"""
        name = name.upper()
        if name not in self._categories:
            self._categories[name] = description
            self._save_custom_categories()
            return True
        return False

    def get_category(self, name: str) -> str:
        """Get category by name"""
        return name.upper() if name.upper() in self._categories else "CONTEXTUAL"

    def list_categories(self) -> Dict[str, str]:
        """List all available categories"""
        return self._categories

    def category_exists(self, name: str) -> bool:
        """Check if category exists"""
        return name.upper() in self._categories