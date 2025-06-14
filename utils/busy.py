# utils/busy.py

import re
from llm import manager

BUSY_KEYWORDS = {
    "busy", "stop", "occupied", "not now", "can't talk", "later", "do not disturb",
    "need to focus", "working", "in a meeting", "no time", "leave me alone", "be quiet"
}
RESUME_KEYWORDS = {
    "resume", "continue", "i'm back", "i am back", "free now", "can talk", "available",
    "done", "let's chat", "talk to me", "i'm here", "i am here", "let's continue"
}

def detect_busy_intent(text):
    """Returns True if text indicates user wants to pause/stop."""
    text = text.lower()
    for phrase in BUSY_KEYWORDS:
        if phrase in text:
            return True
    if re.search(r'\bbusy\b|\bstop\b|\boccupied\b', text):
        return True
    return False

def detect_resume_intent(text):
    """Returns True if text indicates user wants to resume."""
    text = text.lower()
    for phrase in RESUME_KEYWORDS:
        if phrase in text:
            return True
    if re.search(r'\bresume\b|\bcontinue\b|\bback\b|\bfree\b', text):
        return True
    return False

def classify_intent_llm(user_message):
    """
    Uses the LLM to classify user intent as 'busy', 'resume', or 'none'.
    Always returns one of those three strings.
    """
    prompt = (
        "Classify the user's intent as one of: [busy, resume, none]. "
        "Only output the label (busy, resume, or none), nothing else. "
        "User message: " + user_message
    )
    result = manager.ask_llm(prompt, temperature=0, max_tokens=10)
    intent = result.strip().lower()

    if "busy" in intent:
        return "busy"
    elif "resume" in intent or "continue" in intent or "back" in intent or "free" in intent:
        return "resume"
    elif "none" in intent:
        return "none"
    else:
        # Fallback: if LLM hallucinates, treat as "none"
        return "none"