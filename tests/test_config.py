import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timezone

from config import Config


def test_city_local_date_uses_dst_aware_timezone(monkeypatch):
    cfg = Config()
    # July 2, 2026 04:30 UTC = July 2 00:30 in New York (EDT).
    monkeypatch.setattr(cfg, "utc_now", lambda: datetime(2026, 7, 2, 4, 30, tzinfo=timezone.utc))

    assert cfg.city_local_date("New York") == date(2026, 7, 2)


def test_market_day_completion_respects_dst_boundary(monkeypatch):
    cfg = Config()
    market_date = date(2026, 7, 1)

    # July 2, 2026 10:30 UTC = 06:30 in New York (EDT), so the market day is complete.
    monkeypatch.setattr(cfg, "utc_now", lambda: datetime(2026, 7, 2, 10, 30, tzinfo=timezone.utc))
    assert cfg.is_market_day_complete("New York", market_date) is True

    # Thirty minutes earlier is 05:30 EDT, which is still before the completion cutoff.
    monkeypatch.setattr(cfg, "utc_now", lambda: datetime(2026, 7, 2, 9, 30, tzinfo=timezone.utc))
    assert cfg.is_market_day_complete("New York", market_date) is False
