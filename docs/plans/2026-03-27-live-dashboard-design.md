# Live Trading Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Real-time terminal-aesthetic web dashboard showing live positions, spreads, forecasts, exposure, and trade history for the Polymarket weather trading bot.

**Architecture:** Single-file FastAPI server (`dashboard.py`) reads a JSON state snapshot (`/tmp/bot_state.json`) written by `main.py` each cycle. Single HTML file (`dashboard.html`) polls `/api/state` every 3s. No database, no build step, no npm.

**Tech Stack:** FastAPI + uvicorn, vanilla JS, JetBrains Mono font, CSS grid layout.

---

### Task 1: Add state snapshot export to main.py

**Files:**
- Modify: `main.py`

**Step 1: Add state export function**

At the top of `main.py`, add the import:

```python
import json
```

After `run_scan_cycle()` (before `async def main()`), add:

```python
STATE_FILE = "/tmp/bot_state.json"

def export_dashboard_state(
    tracker: PositionTracker,
    engine: DecisionEngine,
    resolution_tracker: ResolutionTracker,
    cycle_count: int,
    start_time: datetime,
    mode: str,
):
    """Write JSON snapshot for the dashboard to read."""
    try:
        orders = list(tracker._orders.values())
        positions = [
            {
                "order_id": o.order_id,
                "city": o.city,
                "date": o.market_date.isoformat(),
                "bucket": o.outcome_label,
                "side": o.side,
                "size": o.intended_size_usd,
                "entry_price": o.limit_price,
                "current_price": o.avg_fill_price or o.limit_price,
                "filled_usd": o.filled_size_usd,
                "status": o.status.value,
            }
            for o in orders if not o.status.value in ("cancelled", "failed")
        ]
        pending = [
            {
                "order_id": o.order_id,
                "city": o.city,
                "date": o.market_date.isoformat(),
                "bucket": o.outcome_label,
                "side": o.side,
                "size": o.intended_size_usd,
                "limit_price": o.limit_price,
                "age_seconds": o.age_seconds,
                "status": o.status.value,
            }
            for o in orders if o.status == OrderStatus.PENDING
        ]

        # Read recent trades from CSV
        recent_trades = []
        trade_log = Path("logs/trades.csv")
        if trade_log.exists():
            with open(trade_log) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in rows[-20:]:
                    recent_trades.append({
                        "timestamp": row.get("timestamp", ""),
                        "city": row.get("city", ""),
                        "date": row.get("market_date", ""),
                        "bucket": row.get("outcome", ""),
                        "side": row.get("side", ""),
                        "price": row.get("price_limit", ""),
                        "size": row.get("intended_usd", ""),
                        "ev": row.get("ev", ""),
                        "edge": row.get("edge", ""),
                        "p_true": row.get("p_true", ""),
                        "market_price": row.get("market_price", ""),
                        "status": row.get("fill_status", ""),
                        "is_maker": row.get("is_maker", ""),
                    })

        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live" if cfg.is_live else "dry-run",
            "cycle": cycle_count,
            "uptime_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "bankroll": cfg.bankroll,
            "daily_pnl": resolution_tracker.state.total_pnl,
            "daily_loss_cap_used": abs(engine.daily_pnl) / cfg.daily_loss_cap if cfg.daily_loss_cap > 0 else 0,
            "total_exposure": tracker.total_exposure,
            "realized_exposure": tracker.realized_exposure,
            "pending_exposure": tracker.pending_exposure,
            "positions": positions,
            "pending_orders": pending,
            "recent_trades": recent_trades,
            "exposure_by_city": {},
            "resolution": {
                "total_pnl": resolution_tracker.state.total_pnl,
                "wins": resolution_tracker.state.total_wins,
                "losses": resolution_tracker.state.total_losses,
                "win_rate": resolution_tracker.state.win_rate,
                "trade_count": resolution_tracker.state.trade_count,
            },
            "adverse_selection": {
                "instant_fill_rate": tracker.adverse_selection_rate,
                "avg_fill_speed": tracker.avg_fill_speed,
            },
            "fill_stats": {
                "total": tracker._total_fills,
                "active": tracker.active_order_count,
                "filled": tracker.filled_order_count,
            },
        }

        # Compute per-city exposure
        city_exp: Dict[str, float] = {}
        for o in orders:
            if not o.is_terminal:
                city_exp[o.city] = city_exp.get(o.city, 0) + o.filled_size_usd + o.unfilled_usd
        state["exposure_by_city"] = city_exp

        # Atomic write: write to tmp then rename
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

    except Exception as e:
        logger.warning(f"Dashboard state export failed: {e}")
```

**Step 2: Add `import os` and `from pathlib import Path` to main.py imports if not present, and import OrderStatus**

Add near the top:
```python
import os
from pathlib import Path
from position_tracker import PositionTracker, OrderStatus
```

(Note: PositionTracker is already imported; just add OrderStatus to the existing import.)

**Step 3: Call export at the end of each cycle in `main()`**

In the `main()` function, record `start_time` before the while loop:
```python
start_time = datetime.now(timezone.utc)
```

After the `run_scan_cycle()` call and before the sleep/wait, add:
```python
export_dashboard_state(
    tracker, engine, resolution_tracker,
    cycle_count, start_time, cfg.mode,
)
```

**Step 4: Verify manually**

Run: `python main.py` briefly, then `cat /tmp/bot_state.json | python -m json.tool`
Expected: Valid JSON with all fields populated.

**Step 5: Commit**

```bash
git add main.py
git commit -m "feat: export dashboard state snapshot each cycle"
```

---

### Task 2: Create FastAPI dashboard backend

**Files:**
- Create: `dashboard.py`

**Step 1: Write the dashboard server**

```python
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
```

**Step 2: Add fastapi and uvicorn to requirements.txt**

Append:
```
fastapi>=0.110.0
uvicorn>=0.29.0
```

**Step 3: Commit**

```bash
git add dashboard.py requirements.txt
git commit -m "feat: add FastAPI dashboard backend"
```

---

### Task 3: Create the terminal-aesthetic dashboard HTML

**Files:**
- Create: `dashboard.html`

This is the largest task. The HTML file is self-contained: embedded CSS + JS, no build step.

**Step 1: Write the full dashboard.html**

Key design decisions:
- Font: JetBrains Mono from Google Fonts
- Colors: #0a0a0a background, #00ff41 green, #ffb000 amber, #ff4444 red, #888 dim
- Layout: CSS grid — top status bar, 3-column main area, bottom exposure bar
- Polling: `setInterval(fetchState, 3000)` calling `/api/state`
- Stale detection: if `timestamp` is >5 minutes old, show STALE warning
- All numbers formatted to 2-3 decimal places
- Positions table: city, date, bucket, side, size, entry price, current price, spread (current - entry), status, color-coded PnL
- Forecast panel: shows edge, p_true vs market_price, confidence per position
- Recent trades: scrollable list of last 20
- Bottom: per-city exposure bars, loss cap meter, adverse selection stats

The full HTML/CSS/JS will be written in one file (~400 lines).

**Step 2: Commit**

```bash
git add dashboard.html
git commit -m "feat: terminal-aesthetic live dashboard UI"
```

---

### Task 4: Add systemd service for the dashboard

**Files:**
- Create: `dashboard.service` (systemd unit file for reference)

**Step 1: Write the service file**

```ini
[Unit]
Description=Weather Bot Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bot
ExecStart=/root/bot/venv/bin/uvicorn dashboard:app --host 0.0.0.0 --port 8050
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Step 2: Commit**

```bash
git add dashboard.service
git commit -m "feat: add systemd service for dashboard"
```

---

### Task 5: End-to-end test

**Step 1: Write a mock bot_state.json for testing**

Create a script that writes sample data to `/tmp/bot_state.json` so the dashboard can be tested without the bot running.

**Step 2: Start the dashboard locally**

```bash
pip install fastapi uvicorn
uvicorn dashboard:app --host 0.0.0.0 --port 8050
```

**Step 3: Open browser to http://localhost:8050 and verify:**
- Top bar shows bot status, uptime, bankroll, exposure, PnL
- Position table renders with color-coded rows
- Recent trades list scrolls
- Bottom exposure bars render
- STALE warning appears if data is old
- Auto-refreshes every 3s

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: live trading dashboard v1 complete"
```
