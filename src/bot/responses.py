def get_response(message: str) -> str:
    """Generate responses based on input message"""
    message = message.lower().strip()
    
    responses = {
        "hello": "Hello! I'm Curie, your friendly AI assistant!",
        "how are you": "I'm functioning perfectly! Thanks for asking!",
        "what can you do": "Currently, I'm in my early stages. I can chat with you and learn new things!",
    }
    
    return responses.get(message, "I'm still learning how to respond to that!")