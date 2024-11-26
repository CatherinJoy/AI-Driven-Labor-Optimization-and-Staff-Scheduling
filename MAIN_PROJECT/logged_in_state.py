import os
import json

STATE_FILE = "logged_in_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    else:
        return {"logged_in": False, "username": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)