# llm_manager.py

import os
from dotenv import load_dotenv

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import config

# Load environment variables from .env
load_dotenv()

# Parse models from .env (comma-separated list)
models_env = os.getenv("LLM_MODELS", "")
AVAILABLE_MODELS = [m.strip() for m in models_env.split(",") if m.strip()]

# Fallback default if nothing set in .env
DEFAULT_LLAMA_MODEL = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load LLM config at module level (or pass in as needed)
llm_config = config.load_config().get('llm', {})

# Cache for loaded llama models
llama_models_cache = {}

def ask_llm(prompt, model_name=None):
    """
    Send a prompt to the LLM and return its response.
    Optionally specify model_name (must be in AVAILABLE_MODELS).
    """
    provider = llm_config.get('provider', 'llama.cpp')

    if provider == 'openai':
        # Example OpenAI API usage (uncomment and fill in if you want)
        """
        import openai
        openai.api_key = llm_config['openai_api_key']
        response = openai.ChatCompletion.create(
            model=llm_config.get('model', 'gpt-3.5-turbo'),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content'].strip()
        """
        return f"[OpenAI simulated response to]: {prompt}"

    elif provider == 'llama.cpp':
        if Llama is None:
            return "[llama-cpp-python not installed. Please install it with 'pip install llama-cpp-python']"

        # Decide which model filename to use
        selected_model = model_name or llm_config.get('model_path') or DEFAULT_LLAMA_MODEL
        if selected_model not in AVAILABLE_MODELS:
            return f"[Model not found in AVAILABLE_MODELS: {selected_model}]"
        model_path = os.path.join("models", selected_model)

        if not os.path.exists(model_path):
            return f"[Model file not found: {model_path}]"

        # Lazy-load and cache each model separately
        if selected_model not in llama_models_cache:
            try:
                llama_models_cache[selected_model] = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4
                )
            except Exception as e:
                return f"[Error loading model: {e}]"
        llama_model = llama_models_cache[selected_model]

        # Run inference
        try:
            result = llama_model(
                prompt,
                max_tokens=512,
                stop=["</s>", "User:", "user:"],
                temperature=llm_config.get("temperature", 0.7)
            )
            if isinstance(result, dict) and "choices" in result:
                return result["choices"][0]["text"].strip()
            elif hasattr(result, "choices"):
                return result.choices[0].text.strip()
            else:
                return str(result)
        except Exception as e:
            return f"[Error during inference: {e}]"

    else:
        return "[Error: Unsupported LLM provider]"


def get_available_models():
    """
    Returns the list of available model filenames.
    """
    return AVAILABLE_MODELS