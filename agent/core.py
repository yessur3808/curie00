# agent/core.py

import memory_manager
import llm_manager
import json

class Agent:
    def __init__(self, persona=None, max_history=5):
        self.persona = persona
        self.max_history = max_history
        memory_manager.init_memory()

    def extract_user_facts(self, user_message):
        """
        Use LLM to extract preferences, interests, traits, etc. from user input.
        Returns a dict of facts.
        """
        prompt = (
            "Extract any preferences, likes, interests, or personality traits about the user from the following message. "
            "Return them as a JSON dictionary of key:value pairs. If nothing can be extracted, return {}.\n"
            f"User message: {user_message}\n"
            "Extracted (JSON):"
        )
        result = llm_manager.ask_llm(prompt, temperature=0.2, max_tokens=100)
        try:
            facts = json.loads(result.strip())
            if isinstance(facts, dict):
                return facts
        except Exception:
            pass
        return {}

    def handle_message(self, message, internal_id=None, chat_context=None):
        if not internal_id:
            raise ValueError("internal_id is required for conversation tracking.")

        memory_manager.save_conversation(internal_id, "user", message)

        # --- Extract and store user facts in MongoDB ---
        facts = self.extract_user_facts(message)
        if facts:
            memory_manager.update_user_profile(internal_id, facts)

        # Load recent user profile from MongoDB
        user_profile = memory_manager.get_user_profile(internal_id)

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
            # --- Inject user facts into the persona prompt ---
            if user_profile:
                conversation += "Here are some things I know about the user so far:\n"
                for k, v in user_profile.items():
                    conversation += f"- {k}: {v}\n"
                conversation += "Use these facts to make your response more personal and relevant.\n"

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

    def generate_small_talk(self, internal_id, chat_context=None):
        """
        Generate a natural, context-aware small talk question or comment,
        in Curie's style, using the LLM and the user's stored profile.
        """
        persona = self.persona
        recent_history = memory_manager.load_recent_conversation(internal_id, limit=6)
        user_profile = memory_manager.get_user_profile(internal_id)

        prompt = (
            f"{persona['system_prompt']}\n"
            "You are in a friendly conversation. "
            "Please generate a brief, friendly, and natural small talk question or comment, or display curiosity about certain topics or the user, "
            "in the style of Curie (occasionally using simple French phrases), that helps get to know the user. "
            "Do not repeat previous questions. Be creative and context-aware.\n"
        )
        if user_profile:
            prompt += "Here are some things you know about the user:\n"
            for k, v in user_profile.items():
                prompt += f"- {k}: {v}\n"
        prompt += "Here is the recent chat history (user and assistant):\n"
        for role, msg in recent_history:
            prompt += f"{role.capitalize()}: {msg}\n"
        prompt += "Curie (small talk):"

        small_talk = llm_manager.ask_llm(prompt, temperature=0.9, max_tokens=60)
        return small_talk.strip()