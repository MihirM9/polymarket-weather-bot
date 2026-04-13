import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import date, datetime, timezone

from trading.decision import TradeSignal
from trading.dry_run import OrderbookSnapshot, SimulatedFill
from trading.execution import OrderExecutor, TradeLogger
from trading.positions import OpenOrder, OrderStatus, PositionTracker


def test_trade_logger_keeps_recent_trades():
    logger = TradeLogger()
    signal = TradeSignal(
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        token_id="tok1",
        side="BUY",
        p_true=0.55,
        market_price=0.40,
        ev=0.10,
        edge=0.15,
        kelly_fraction=0.03,
        position_size_usd=5.0,
        price_limit=0.41,
        rationale="test signal",
    )
    order = OpenOrder(
        order_id="ord1",
        token_id="tok1",
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        side="BUY",
        intended_size_usd=5.0,
        limit_price=0.41,
        submitted_at=datetime.now(timezone.utc),
        status=OrderStatus.FILLED,
        filled_size_usd=5.0,
        avg_fill_price=0.41,
    )

    logger.log_trade(signal, order, True, slippage=0.0, fill_ratio=1.0, is_maker=True, book_depth=10.0)

    recent = logger.recent_trades
    assert len(recent) == 1
    assert recent[0]["bucket"] == "80-81°F"


def test_order_executor_uses_simulated_fill_metadata(monkeypatch):
    monkeypatch.setattr(PositionTracker, "_load_state", lambda self: None)
    tracker = PositionTracker()
    executor = OrderExecutor(tracker)

    async def fake_fetch_orderbook(session, token_id):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[(0.40, 100.0)],
            asks=[(0.42, 100.0)],
            timestamp=datetime.now(timezone.utc),
        )

    def fake_simulate_fill(snapshot, side, intended_size_usd, limit_price, is_maker=True):
        return SimulatedFill(
            filled_size_usd=intended_size_usd,
            filled_shares=intended_size_usd / limit_price,
            avg_fill_price=limit_price,
            slippage=0.0,
            fill_ratio=1.0,
            is_maker=True,
            estimated_fill_cycles=0,
        )

    monkeypatch.setattr(executor.simulator, "fetch_orderbook", fake_fetch_orderbook)
    monkeypatch.setattr(executor.simulator, "simulate_fill", fake_simulate_fill)

    signal = TradeSignal(
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        token_id="tok1",
        side="BUY",
        p_true=0.55,
        market_price=0.40,
        ev=0.10,
        edge=0.15,
        kelly_fraction=0.03,
        position_size_usd=5.0,
        price_limit=0.41,
        rationale="test signal",
    )

    order = asyncio.run(executor.execute_signal(signal))

    assert order is not None
    assert order.simulated_is_maker is True
    assert order.simulated_fill_ratio == 1.0
    assert tracker.total_exposure >= 5.0


def test_poll_fills_ignores_malformed_clob_payload(monkeypatch):
    monkeypatch.setattr(PositionTracker, "_load_state", lambda self: None)
    tracker = PositionTracker()

    class BadClient:
        def get_order(self, order_id):
            return {"status": "filled", "size_matched": "not-a-number"}

        def cancel(self, order_id):
            raise AssertionError("cancel should not be called")

    tracker.set_clob_client(BadClient())
    order = OpenOrder(
        order_id="ord-bad",
        token_id="tok1",
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        side="BUY",
        intended_size_usd=5.0,
        limit_price=0.41,
        submitted_at=datetime.now(timezone.utc),
        status=OrderStatus.PENDING,
    )
    tracker.register_order(order)

    changed = asyncio.run(tracker.poll_fills())

    assert changed == 0
    assert tracker._orders["ord-bad"].status == OrderStatus.PENDING
