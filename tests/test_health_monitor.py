import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.health import HealthMonitor


def test_health_monitor_triggers_fail_safe_on_consecutive_cycle_failures():
    monitor = HealthMonitor()

    monitor.record_cycle_failure("one")
    monitor.record_cycle_failure("two")
    assert monitor.fail_safe_reason() is None

    monitor.record_cycle_failure("three")
    reason = monitor.fail_safe_reason()
    assert reason is not None
    assert "Consecutive scan failures" in reason


def test_health_monitor_resets_dashboard_export_failure_counter():
    monitor = HealthMonitor()

    monitor.record_dashboard_export(False)
    monitor.record_dashboard_export(False)
    assert monitor.fail_safe_reason() is None

    monitor.record_dashboard_export(True)
    assert monitor.fail_safe_reason() is None


def test_health_monitor_triggers_fail_safe_on_repeated_dashboard_export_failure():
    monitor = HealthMonitor()

    monitor.record_dashboard_export(False)
    monitor.record_dashboard_export(False)
    monitor.record_dashboard_export(False)

    reason = monitor.fail_safe_reason()
    assert reason is not None
    assert "Dashboard state export failed" in reason
