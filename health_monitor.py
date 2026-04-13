"""
health_monitor.py — Runtime health checks and fail-safe shutdown rules
======================================================================
Tracks critical signals that should stop trading before the bot degrades into
silent bad behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class HealthSnapshot:
    consecutive_cycle_failures: int = 0
    dashboard_export_failures: int = 0
    last_successful_cycle_at: Optional[datetime] = None
    last_failure_reason: str = ""


class HealthMonitor:
    MAX_CONSECUTIVE_CYCLE_FAILURES = 3
    MAX_DASHBOARD_EXPORT_FAILURES = 3

    def __init__(self) -> None:
        self.state = HealthSnapshot()

    def record_cycle_success(self) -> None:
        self.state.consecutive_cycle_failures = 0
        self.state.last_successful_cycle_at = datetime.now(timezone.utc)
        self.state.last_failure_reason = ""

    def record_cycle_failure(self, reason: str) -> None:
        self.state.consecutive_cycle_failures += 1
        self.state.last_failure_reason = reason

    def record_dashboard_export(self, ok: bool) -> None:
        if ok:
            self.state.dashboard_export_failures = 0
        else:
            self.state.dashboard_export_failures += 1

    def fail_safe_reason(self) -> Optional[str]:
        if self.state.consecutive_cycle_failures >= self.MAX_CONSECUTIVE_CYCLE_FAILURES:
            return (
                f"Consecutive scan failures reached {self.state.consecutive_cycle_failures}: "
                f"{self.state.last_failure_reason}"
            )
        if self.state.dashboard_export_failures >= self.MAX_DASHBOARD_EXPORT_FAILURES:
            return (
                f"Dashboard state export failed {self.state.dashboard_export_failures} times in a row"
            )
        return None
