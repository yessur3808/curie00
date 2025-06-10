import llm_manager

class Agent:
    def __init__(self, persona=None):
        self.persona = persona  # You can load or pass in a persona dict

    def handle_message(self, message, user_id=None, chat_context=None):
        """
        Handles incoming messages from the connector.
        For Phase 1, just send the message to the LLM and return the reply.
        """
        # prepend persona/system prompt here
        prompt = message
        if self.persona and self.persona.get("system_prompt"):
            prompt = f"{self.persona['system_prompt']}\nUser: {message}"

        # Call the LLM 
        response = llm_manager.ask_llm(prompt)

        return response