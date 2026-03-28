"""
dashboard.py — Live trading dashboard server
=============================================
Lightweight FastAPI server that reads bot_state.json and serves
a terminal-aesthetic dashboard. Run alongside the bot on the same box.

Usage: uvicorn dashboard:app --host 0.0.0.0 --port 8050
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Weather Bot Dashboard")

STATE_FILE = os.getenv("BOT_STATE_FILE", "/tmp/bot_state.json")
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML.read_text()


@app.get("/api/state", response_class=JSONResponse)
async def get_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Bot state not available yet", "timestamp": None}
    except json.JSONDecodeError:
        return {"error": "State file corrupted", "timestamp": None}
