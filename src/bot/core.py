class CurieBot:
    def __init__(self):
        self.name = "Curie"
        self.version = "0.1.0"

    def initialize(self):
        """Initialize the bot and its components"""
        return f"{self.name} v{self.version} initialized successfully!"

    def process_message(self, message: str) -> str:
        """Process incoming messages and return appropriate responses"""
        from .responses import get_response
        return get_response(message)