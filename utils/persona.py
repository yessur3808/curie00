# utils/persona.py

import os
import json

def load_persona(filename='persona.json', assets_dir='assets'):
    persona_path = os.path.join(assets_dir, filename)
    if os.path.isfile(persona_path):
        with open(persona_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}