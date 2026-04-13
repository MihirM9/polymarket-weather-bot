"""
dashboarding.simulate — Simulate bot state updates for dashboard testing.

Usage:
    python3 -m dashboarding.simulate
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, cast

STATE_FILE = "/tmp/bot_state.json"

# Base positions — prices will drift each tick
POSITIONS = [
    {"order_id": "live-abc123", "city": "Miami", "date": "2026-03-28", "bucket": "80-84", "side": "BUY", "size": 8.50, "entry_price": 0.420, "filled_usd": 8.50, "status": "filled"},
    {"order_id": "live-def456", "city": "Chicago", "date": "2026-03-28", "bucket": "45-49", "side": "BUY", "size": 6.20, "entry_price": 0.350, "filled_usd": 6.20, "status": "filled"},
    {"order_id": "live-ghi789", "city": "Dallas", "date": "2026-03-28", "bucket": "65-69", "side": "SELL", "size": 7.00, "entry_price": 0.550, "filled_usd": 7.00, "status": "filled"},
    {"order_id": "live-jkl012", "city": "Los Angeles", "date": "2026-03-28", "bucket": "70-74", "side": "BUY", "size": 5.50, "entry_price": 0.410, "filled_usd": 0.00, "status": "pending"},
    {"order_id": "live-mno345", "city": "New York", "date": "2026-03-28", "bucket": "55-59", "side": "BUY", "size": 9.00, "entry_price": 0.380, "filled_usd": 9.00, "status": "filled"},
]

TRADE_TEMPLATES = [
    {"city": "Miami", "bucket": "80-84", "side": "BUY"},
    {"city": "Chicago", "bucket": "45-49", "side": "BUY"},
    {"city": "Dallas", "bucket": "65-69", "side": "SELL"},
    {"city": "Houston", "bucket": "75-79", "side": "BUY"},
    {"city": "New York", "bucket": "55-59", "side": "BUY"},
    {"city": "Los Angeles", "bucket": "70-74", "side": "BUY"},
]

start_time = datetime.now(timezone.utc)
cycle = 0
pnl = 12.35
wins = 8
losses = 3
trades_log: list[dict[str, Any]] = []


def drift_price(base, magnitude=0.015):
    return round(max(0.01, min(0.99, base + random.uniform(-magnitude, magnitude))), 3)


def build_state():
    global cycle, pnl, wins, losses, trades_log

    cycle += 1
    now = datetime.now(timezone.utc)
    uptime = (now - start_time).total_seconds()

    # Drift current prices
    positions: list[dict[str, Any]] = []
    for p in POSITIONS:
        pos = cast(dict[str, Any], dict(p))
        pos["current_price"] = drift_price(p["entry_price"])
        positions.append(pos)

    # Occasionally flip pending -> filled
    if positions[3]["status"] == "pending" and random.random() < 0.1:
        positions[3]["status"] = "filled"
        positions[3]["filled_usd"] = positions[3]["size"]
        POSITIONS[3]["status"] = "filled"
        POSITIONS[3]["filled_usd"] = POSITIONS[3]["size"]

    # Occasionally add a trade
    if random.random() < 0.2:
        tmpl = random.choice(TRADE_TEMPLATES)
        price = round(random.uniform(0.25, 0.65), 4)
        size = round(random.uniform(3.0, 10.0), 2)
        edge = round(random.uniform(0.05, 0.20), 4)
        ev = round(random.uniform(0.08, 0.22), 4)
        p_true = round(price + edge, 4)
        trade = {
            "timestamp": now.isoformat(),
            "city": tmpl["city"],
            "date": "2026-03-28",
            "bucket": tmpl["bucket"],
            "side": tmpl["side"],
            "price": f"{price:.4f}",
            "size": f"{size:.2f}",
            "ev": f"{ev:.4f}",
            "edge": f"{edge:.4f}",
            "p_true": f"{p_true:.4f}",
            "market_price": f"{price:.4f}",
            "status": random.choice(["filled", "filled", "pending"]),
            "is_maker": str(random.random() < 0.6),
        }
        trades_log.append(trade)
        trades_log[:] = trades_log[-20:]

    # Drift PnL
    pnl_delta = random.uniform(-0.50, 0.80)
    pnl += pnl_delta
    if pnl_delta > 0:
        wins += 1 if random.random() < 0.3 else 0
    else:
        losses += 1 if random.random() < 0.3 else 0

    total_exposure = sum(float(p["size"]) for p in positions if p["status"] != "cancelled")
    realized = sum(float(p["filled_usd"]) for p in positions)
    pending_exp = total_exposure - realized
    total_deployed = realized + cycle * random.uniform(2.0, 5.0)  # grows over time

    city_exp: dict[str, float] = {}
    for p in positions:
        if p["status"] not in ("cancelled", "failed"):
            city = str(p["city"])
            size_value: Any = p["size"]
            city_size = float(size_value) if isinstance(size_value, (int, float, str)) else 0.0
            city_exp[city] = city_exp.get(city, 0.0) + city_size

    trade_count = wins + losses
    state = {
        "timestamp": now.isoformat(),
        "mode": "live",
        "cycle": cycle,
        "uptime_seconds": uptime,
        "bankroll": 500.0,
        "daily_pnl": round(pnl, 2),
        "daily_loss_cap_used": round(abs(pnl) / 50.0, 4),
        "total_exposure": round(total_exposure, 2),
        "total_deployed": round(total_deployed, 2),
        "realized_exposure": round(realized, 2),
        "pending_exposure": round(max(0, pending_exp), 2),
        "positions": positions,
        "pending_orders": [p for p in positions if p["status"] == "pending"],
        "recent_trades": trades_log[-20:],
        "exposure_by_city": city_exp,
        "resolution": {
            "total_pnl": round(pnl, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(1, trade_count), 3),
            "trade_count": trade_count,
        },
        "adverse_selection": {
            "instant_fill_rate": round(random.uniform(0.03, 0.12), 3),
            "avg_fill_speed": round(random.uniform(20, 80), 1),
        },
        "fill_stats": {
            "total": 42 + cycle,
            "active": sum(1 for p in positions if p["status"] == "pending"),
            "filled": sum(1 for p in positions if p["status"] == "filled"),
        },
    }
    return state


def main():
    print(f"Simulating bot state → {STATE_FILE}")
    print("Dashboard at http://localhost:8050")
    print("Ctrl+C to stop\n")

    while True:
        state = build_state()
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

        print(
            f"  cycle={state['cycle']:>4}  "
            f"uptime={state['uptime_seconds']:>6.0f}s  "
            f"pnl=${state['daily_pnl']:>+7.2f}  "
            f"exposure=${state['total_exposure']:>6.2f}  "
            f"trades={len(state['recent_trades'])}",
            flush=True,
        )
        time.sleep(3)


if __name__ == "__main__":
    main()
