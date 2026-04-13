import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import date, datetime, timezone

from trading import positions as positions_module
from trading.positions import OpenOrder, OrderStatus, PositionTracker


def test_apply_dry_run_fill_tick_moves_exposure(monkeypatch, tmp_path):
    monkeypatch.setattr(positions_module, "POSITION_LOG", tmp_path / "positions.csv")
    monkeypatch.setattr(positions_module, "TRACKER_STATE_FILE", tmp_path / "tracker_state.json")
    monkeypatch.setattr(PositionTracker, "_load_state", lambda self: None)

    tracker = PositionTracker()
    order = OpenOrder(
        order_id="dry-123",
        token_id="tok",
        market_id="mkt",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        side="BUY",
        intended_size_usd=10.0,
        limit_price=0.25,
        submitted_at=datetime.now(timezone.utc),
        status=OrderStatus.PENDING,
    )
    tracker.register_order(order)

    order.filled_size_usd = 6.0
    order.filled_shares = 24.0
    order.avg_fill_price = 0.25
    order.status = OrderStatus.PARTIAL

    applied = tracker.apply_dry_run_fill_tick([order])

    assert applied == 1
    assert tracker.realized_exposure == 6.0
    assert tracker.pending_exposure == 4.0


def test_position_tracker_restores_saved_state(monkeypatch, tmp_path):
    state_file = tmp_path / "tracker_state.json"
    monkeypatch.setattr(positions_module, "TRACKER_STATE_FILE", state_file)
    monkeypatch.setattr(positions_module, "POSITION_LOG", tmp_path / "positions.csv")
    today = date.today().isoformat()
    state_file.write_text(json.dumps({
        "today": today,
        "daily_realized": 7.5,
        "daily_pending": 2.5,
        "total_fills": 1,
        "instant_fill_count": 0,
        "fill_speeds": [12.0],
        "orders": {
            "ord1": {
                "order_id": "ord1",
                "token_id": "tok1",
                "market_id": "mkt1",
                "city": "Miami",
                "market_date": today,
                "outcome_label": "80-81°F",
                "side": "BUY",
                "intended_size_usd": 10.0,
                "limit_price": 0.4,
                "submitted_at": f"{today}T12:00:00+00:00",
                "status": "partial",
                "filled_size_usd": 7.5,
                "filled_shares": 18.75,
                "avg_fill_price": 0.4,
            }
        },
    }))

    tracker = PositionTracker()

    assert tracker.realized_exposure == 7.5
    assert tracker.pending_exposure == 2.5
    assert tracker.active_order_count == 1
