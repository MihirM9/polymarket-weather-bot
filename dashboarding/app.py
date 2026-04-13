"""
dashboarding.app — Live trading dashboard server.

Usage:
    uvicorn dashboarding.app:app --host 0.0.0.0 --port 8050
"""

import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

STATE_FILE = os.getenv("BOT_STATE_FILE", "/tmp/bot_state.json")
DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "dashboard.html"


async def index(request):
    return HTMLResponse(DASHBOARD_HTML.read_text())


async def get_state(request):
    try:
        with open(STATE_FILE) as f:
            return JSONResponse(json.load(f))
    except FileNotFoundError:
        return JSONResponse({"error": "Bot state not available yet", "timestamp": None})
    except json.JSONDecodeError:
        return JSONResponse({"error": "State file corrupted", "timestamp": None})


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/state", get_state),
    ]
)
