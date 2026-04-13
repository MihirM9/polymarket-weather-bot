import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timezone

from position_tracker import OpenOrder, OrderStatus, PositionTracker


def test_apply_dry_run_fill_tick_moves_exposure(monkeypatch, tmp_path):
    monkeypatch.setattr("position_tracker.POSITION_LOG", tmp_path / "positions.csv")
    monkeypatch.setattr("position_tracker.TRACKER_STATE_FILE", tmp_path / "tracker_state.json")
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
