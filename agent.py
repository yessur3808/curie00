# agent.py

import memory_manager
import llm_manager

class Agent:
    def __init__(self, persona=None, max_history=5):
        self.persona = persona
        self.max_history = max_history
        memory_manager.init_memory()

    def handle_message(self, message, internal_id=None, chat_context=None):
        if not internal_id:
            raise ValueError("internal_id is required for conversation tracking.")

        memory_manager.save_conversation(internal_id, "user", message)

        # Load recent conversation history from Postgres
        history = memory_manager.load_recent_conversation(internal_id, limit=self.max_history * 2)
        conversation = ""
        if self.persona and self.persona.get("system_prompt"):
            conversation += self.persona["system_prompt"] + "\n"
            conversation += (
                "Occasionally, use simple French words or phrases (like 'oui', 'non', 'mon ami', etc.) as interjections or for extra charm, "
                "in the style of Curie from Fallout 4, but never so much that you confuse an English speaker. "
                "Ensure your overall meaning is always clear.\n"
            )
            conversation += (
                "When you respond, be detailed and thorough, like a friendly, lively, and knowledgeable companion. "
                "Share your reasoning, add a relevant anecdote or fact if appropriate, and try to make your tone warm and personal. "
                "If you can, ask a gentle follow-up question to keep the conversation engaging. "
                "Avoid sounding robotic or detached.\n"
            )
        for role, msg in history:
            if role == "user":
                conversation += f"User: {msg}\n"
            else:
                conversation += f"Curie: {msg}\n"
        conversation += f"User: {message}\nCurie:"

        response = llm_manager.ask_llm(conversation)
        memory_manager.save_conversation(internal_id, "assistant", response)
        return response
    
    
    
    def save_research(self, topic, content, internal_id=None):
        memory_manager.save_research(topic, content, internal_id)

    def search_research(self, topic, internal_id=None):
        return memory_manager.search_research(topic, internal_id)