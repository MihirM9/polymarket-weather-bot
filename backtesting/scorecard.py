"""Backtest Scorecard — metrics, breakdowns, fragility analysis for backtest results.

Standalone module: only stdlib imports (no codebase dependencies).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import date
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """Single simulated trade produced by the backtester."""

    city: str
    target_date: date
    days_out: int              # forecast horizon (0 = same-day)
    side: str                  # "BUY" or "SELL"
    outcome_label: str         # e.g. "72-73°F"
    bucket_low: float
    bucket_high: float
    p_true: float              # model probability
    market_price: float        # Polymarket price at decision time
    ev: float                  # expected value
    edge: float                # p_true - market_price (or similar)
    kelly_fraction: float
    position_size_usd: float
    price_limit: float
    actual_high: float         # observed high temperature
    won: bool
    pnl: float                # realized P&L in USD
    variant: str               # e.g. "realistic", "optimistic"
    regime: str = "normal"     # weather regime at trade time
    price_source: str = "synthetic"  # "real_clob", "real_gamma", or "synthetic"


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class BacktestScorecard:
    """Compute and render performance metrics over a list of BacktestTrades."""

    def __init__(self, trades: List[BacktestTrade]) -> None:
        self.trades = trades

    # -- filtering ----------------------------------------------------------

    def _filter(self, variant: str) -> List[BacktestTrade]:
        """Return trades matching *variant*."""
        return [t for t in self.trades if t.variant == variant]

    # -- scalar metrics -----------------------------------------------------

    def win_rate(self, variant: str) -> float:
        ts = self._filter(variant)
        if not ts:
            return 0.0
        return sum(1 for t in ts if t.won) / len(ts)

    def total_pnl(self, variant: str) -> float:
        return sum(t.pnl for t in self._filter(variant))

    def trade_count(self, variant: str) -> int:
        return len(self._filter(variant))

    def profit_factor(self, variant: str) -> float:
        """Gross wins / gross losses.  Returns inf when no losses."""
        ts = self._filter(variant)
        gross_win = sum(t.pnl for t in ts if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in ts if t.pnl < 0))
        if gross_loss == 0:
            return float("inf")
        return gross_win / gross_loss

    def max_drawdown(self, variant: str) -> float:
        """Peak-to-trough drawdown from sequential PnL series."""
        ts = self._filter(variant)
        if not ts:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in ts:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def avg_drawdown(self, variant: str) -> float:
        """Average of all drawdown periods (peak-to-trough segments)."""
        ts = self._filter(variant)
        if not ts:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        drawdowns: list[float] = []
        in_drawdown = False
        current_dd = 0.0
        for t in ts:
            cumulative += t.pnl
            if cumulative > peak:
                if in_drawdown and current_dd > 0:
                    drawdowns.append(current_dd)
                peak = cumulative
                in_drawdown = False
                current_dd = 0.0
            else:
                dd = peak - cumulative
                if dd > 0:
                    in_drawdown = True
                    current_dd = dd
        # capture trailing drawdown
        if in_drawdown and current_dd > 0:
            drawdowns.append(current_dd)
        return sum(drawdowns) / len(drawdowns) if drawdowns else 0.0

    def avg_ev(self, variant: str) -> float:
        ts = self._filter(variant)
        if not ts:
            return 0.0
        return sum(t.ev for t in ts) / len(ts)

    # -- time-series helpers ------------------------------------------------

    def _daily_pnl_series(self, variant: str) -> List[tuple[date, float]]:
        """Group PnL by target_date, return sorted (date, daily_pnl) pairs."""
        ts = self._filter(variant)
        by_day: Dict[date, float] = defaultdict(float)
        for t in ts:
            by_day[t.target_date] += t.pnl
        return sorted(by_day.items())

    # -- risk-adjusted returns ----------------------------------------------

    def sharpe_ratio(self, variant: str) -> float:
        """Annualized Sharpe ratio (sqrt(365))."""
        series = self._daily_pnl_series(variant)
        if len(series) < 2:
            return 0.0
        pnls = [p for _, p in series]
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(365)

    def sortino_ratio(self, variant: str) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        series = self._daily_pnl_series(variant)
        if len(series) < 2:
            return 0.0
        pnls = [p for _, p in series]
        mean = sum(pnls) / len(pnls)
        downside = [min(p, 0) ** 2 for p in pnls]
        down_dev = math.sqrt(sum(downside) / (len(pnls) - 1))
        if down_dev == 0:
            return 0.0
        return (mean / down_dev) * math.sqrt(365)

    def calmar_ratio(self, variant: str) -> float:
        """Annualized return / max drawdown."""
        series = self._daily_pnl_series(variant)
        if not series:
            return 0.0
        total = sum(p for _, p in series)
        n_days = (series[-1][0] - series[0][0]).days or 1
        annual_return = total * 365 / n_days
        mdd = self.max_drawdown(variant)
        if mdd == 0:
            return float("inf") if annual_return > 0 else 0.0
        return annual_return / mdd

    # -- breakdowns ---------------------------------------------------------

    def _breakdown(self, variant: str, key_fn: Callable[[BacktestTrade], Any]) -> Dict[Any, Dict[str, Any]]:
        """Generic grouping: returns {key: {win_rate, pnl, trades, avg_ev}}."""
        ts = self._filter(variant)
        groups: Dict[Any, List[BacktestTrade]] = defaultdict(list)
        for t in ts:
            groups[key_fn(t)].append(t)
        result: Dict[Any, Dict[str, Any]] = {}
        for key, group in groups.items():
            wins = sum(1 for t in group if t.won)
            result[key] = {
                "win_rate": wins / len(group) if group else 0.0,
                "pnl": sum(t.pnl for t in group),
                "trades": len(group),
                "avg_ev": sum(t.ev for t in group) / len(group) if group else 0.0,
            }
        return result

    def breakdown_by_city(self, variant: str) -> Dict[str, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: t.city)

    def breakdown_by_month(self, variant: str) -> Dict[int, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: t.target_date.month)

    def breakdown_by_horizon(self, variant: str) -> Dict[int, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: t.days_out)

    def breakdown_by_regime(self, variant: str) -> Dict[str, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: t.regime)

    def breakdown_by_side(self, variant: str) -> Dict[str, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: t.side)

    def breakdown_by_source(self, variant: str) -> Dict[str, Dict[str, Any]]:
        return self._breakdown(variant, lambda t: getattr(t, 'price_source', 'synthetic'))

    # -- fragility analysis -------------------------------------------------

    def robustness_score(
        self,
        variant: str,
        sensitivity_variance: float = 0.0,
        leakage_gap: float = 0.0,
    ) -> Tuple[float, List[str]]:
        """Weighted robustness score (0-100) with penalty breakdown.

        Args:
            sensitivity_variance: std dev of Sharpe across parameter sweeps (0 if not run)
            leakage_gap: optimistic_sharpe - realistic_sharpe (measures look-ahead leakage)

        Returns:
            (score, [penalty_descriptions])
        """
        score = 100.0
        penalties: List[str] = []

        # --- Sharpe penalty ---
        sr = self.sharpe_ratio(variant)
        if sr < 1.0:
            score -= 20
            penalties.append(f"Sharpe {sr:.2f} < 1.0: -20")
        elif sr < 1.5:
            score -= 10
            penalties.append(f"Sharpe {sr:.2f} < 1.5: -10")
        elif sr < 2.0:
            score -= 5
            penalties.append(f"Sharpe {sr:.2f} < 2.0: -5")

        # --- Win rate penalty ---
        wr = self.win_rate(variant)
        if wr < 0.55:
            score -= 15
            penalties.append(f"Win rate {wr:.1%} < 55%: -15")
        elif wr < 0.60:
            score -= 10
            penalties.append(f"Win rate {wr:.1%} < 60%: -10")
        elif wr < 0.62:
            score -= 5
            penalties.append(f"Win rate {wr:.1%} < 62%: -5")

        # --- Drawdown penalty (as % of total PnL used as proxy for bankroll) ---
        # Use total_pnl as bankroll proxy; if no profit, use absolute max_dd
        total = self.total_pnl(variant)
        mdd = self.max_drawdown(variant)
        # Express drawdown as fraction of bankroll (approximate via total PnL)
        # If total PnL <= 0, any drawdown is severe
        if total > 0:
            dd_pct = mdd / total
        else:
            dd_pct = 1.0 if mdd > 0 else 0.0

        if dd_pct > 0.20:
            score -= 15
            penalties.append(f"Max DD {dd_pct:.0%} of bankroll > 20%: -15")
        elif dd_pct > 0.15:
            score -= 10
            penalties.append(f"Max DD {dd_pct:.0%} of bankroll > 15%: -10")
        elif dd_pct > 0.10:
            score -= 5
            penalties.append(f"Max DD {dd_pct:.0%} of bankroll > 10%: -5")

        # --- Regime concentration penalty ---
        by_regime = self.breakdown_by_regime(variant)
        total_pnl_abs = total if total != 0 else 1.0
        for regime, stats in by_regime.items():
            regime_pnl_frac = stats["pnl"] / total_pnl_abs if total_pnl_abs != 0 else 0.0
            if stats["trades"] >= 10 and stats["pnl"] < 0:
                score -= 15
                penalties.append(
                    f"Regime '{regime}' negative PnL ${stats['pnl']:.2f} with {stats['trades']} trades: -15"
                )
                break  # only penalize once for worst offender
        else:
            # Check concentration only if no negative-regime penalty applied
            for regime, stats in by_regime.items():
                if total > 0 and stats["pnl"] / total > 0.40:
                    score -= 10
                    penalties.append(
                        f"Regime '{regime}' contributes {stats['pnl'] / total:.0%} of total PnL (>40%): -10"
                    )
                    break

        # --- Month concentration penalty ---
        by_month = self.breakdown_by_month(variant)
        worst_month_pnl = 0.0
        for month, stats in by_month.items():
            if total > 0 and stats["pnl"] / total > 0.35:
                score -= 10
                penalties.append(
                    f"Month {month} contributes {stats['pnl'] / total:.0%} of total PnL (>35%): -10"
                )
                break
        for month, stats in by_month.items():
            if stats["pnl"] < worst_month_pnl:
                worst_month_pnl = stats["pnl"]
        if total > 0 and worst_month_pnl < 0 and abs(worst_month_pnl) / total > 0.05:
            score -= 5
            penalties.append(
                f"Worst month PnL ${worst_month_pnl:.2f} is >{5}% of total PnL: -5"
            )

        # --- Side imbalance penalty ---
        by_side = self.breakdown_by_side(variant)
        n_trades = self.trade_count(variant)
        if n_trades > 0:
            for side, stats in by_side.items():
                if stats["trades"] / n_trades > 0.80:
                    score -= 10
                    penalties.append(
                        f"Side '{side}' is {stats['trades'] / n_trades:.0%} of trades (>80%): -10"
                    )
                    break

        # --- Sensitivity variance penalty ---
        if sensitivity_variance > 0:
            sv_penalty = 5.0 * min(4.0, sensitivity_variance / 0.3)
            score -= sv_penalty
            penalties.append(
                f"Sensitivity variance {sensitivity_variance:.3f} (Sharpe instability): -{sv_penalty:.1f}"
            )

        # --- Look-ahead leakage penalty ---
        if leakage_gap > 1.0:
            score -= 20
            penalties.append(f"Look-ahead leakage gap {leakage_gap:.2f} > 1.0: -20")
        elif leakage_gap > 0.5:
            score -= 10
            penalties.append(f"Look-ahead leakage gap {leakage_gap:.2f} > 0.5: -10")

        # --- Low trade count penalty ---
        if n_trades < 50:
            score -= 10
            penalties.append(f"Only {n_trades} trades (< 50): -10")
        elif n_trades < 100:
            score -= 5
            penalties.append(f"Only {n_trades} trades (< 100): -5")

        # --- Synthetic data circularity penalty ---
        ts = self._filter(variant)
        if ts:
            synthetic_count = sum(1 for t in ts if getattr(t, 'price_source', 'synthetic') == 'synthetic')
            synthetic_pct = synthetic_count / len(ts)
            if synthetic_pct > 0.9:
                score -= 15
                penalties.append(
                    f"Circular risk: {synthetic_pct:.0%} of trades use synthetic prices: -15"
                )
            elif synthetic_pct > 0.5:
                score -= 10
                penalties.append(
                    f"Partial circular risk: {synthetic_pct:.0%} synthetic prices: -10"
                )

        score = max(0.0, score)
        return score, penalties

    @staticmethod
    def robustness_verdict(score: float) -> str:
        """Map robustness score to verdict label."""
        if score >= 80:
            return "ROBUST — strategy validated for paper trading"
        elif score >= 60:
            return "FRAGILE — strategy has weaknesses, investigate before deploying"
        elif score >= 40:
            return "WEAK — strategy likely has structural issues"
        else:
            return "BROKEN — do not deploy"

    def fragility_notes(self, variant: str) -> List[str]:
        """Return list of warning strings for weak segments."""
        notes: list[str] = []

        # Check overall metrics
        wr = self.win_rate(variant)
        sr = self.sharpe_ratio(variant)

        if sr < 1.5:
            notes.append(f"FAIL: Sharpe ratio {sr:.2f} < 1.5 target")
        else:
            notes.append(f"PASS: Sharpe ratio {sr:.2f} >= 1.5")

        if wr < 0.62:
            notes.append(f"FAIL: Win rate {wr:.1%} < 62% target")
        else:
            notes.append(f"PASS: Win rate {wr:.1%} >= 62%")

        # Weak cities
        by_city = self.breakdown_by_city(variant)
        for city, stats in by_city.items():
            if stats["trades"] >= 5 and stats["win_rate"] < 0.50:
                notes.append(f"WEAK CITY: {city} win rate {stats['win_rate']:.1%} ({stats['trades']} trades)")

        # Weak horizons
        by_horizon = self.breakdown_by_horizon(variant)
        for horizon, stats in sorted(by_horizon.items()):
            if stats["trades"] >= 5 and stats["win_rate"] < 0.50:
                notes.append(f"WEAK HORIZON: {horizon}d win rate {stats['win_rate']:.1%} ({stats['trades']} trades)")

        # Weak regimes
        by_regime = self.breakdown_by_regime(variant)
        total = self.total_pnl(variant)
        for regime, stats in by_regime.items():
            if stats["trades"] >= 5 and stats["win_rate"] < 0.50:
                notes.append(f"WEAK REGIME: {regime} win rate {stats['win_rate']:.1%} ({stats['trades']} trades)")
            if total > 0 and stats["pnl"] / total > 0.40:
                notes.append(
                    f"CONCENTRATED REGIME: {regime} contributes {stats['pnl'] / total:.0%} of total PnL"
                )
            if stats["trades"] >= 10 and stats["pnl"] < 0:
                notes.append(
                    f"LOSING REGIME: {regime} PnL ${stats['pnl']:.2f} over {stats['trades']} trades"
                )

        # Month concentration
        by_month = self.breakdown_by_month(variant)
        for month, stats in by_month.items():
            if total > 0 and stats["pnl"] / total > 0.35:
                notes.append(
                    f"CONCENTRATED MONTH: month {month} contributes {stats['pnl'] / total:.0%} of total PnL"
                )
            if stats["trades"] >= 5 and stats["pnl"] < 0:
                notes.append(
                    f"LOSING MONTH: month {month} PnL ${stats['pnl']:.2f} over {stats['trades']} trades"
                )

        # Price source distribution
        ts = self._filter(variant)
        if ts:
            sources: Dict[str, int] = defaultdict(int)
            for t in ts:
                sources[getattr(t, 'price_source', 'synthetic')] += 1
            for src, count in sorted(sources.items()):
                pct = count / len(ts)
                notes.append(f"PRICE SOURCE: {src} = {pct:.0%} ({count} trades)")
            synthetic_pct = sources.get('synthetic', 0) / len(ts)
            if synthetic_pct > 0.9:
                notes.append("WARNING: >90% synthetic prices — results are circular, not trustworthy")

        return notes

    # -- rendering ----------------------------------------------------------

    def render(self, variant: str) -> str:
        """Full terminal scorecard with all sections."""
        lines: list[str] = []
        sep = "=" * 60

        lines.append(sep)
        lines.append(f"  BACKTEST SCORECARD — variant: {variant}")
        lines.append(sep)

        # Summary
        lines.append("")
        lines.append("  SUMMARY")
        lines.append(f"  Trades:         {self.trade_count(variant)}")
        lines.append(f"  Win Rate:       {self.win_rate(variant):.1%}")
        lines.append(f"  Total PnL:      ${self.total_pnl(variant):,.2f}")
        lines.append(f"  Profit Factor:  {self.profit_factor(variant):.2f}")
        lines.append(f"  Avg EV:         {self.avg_ev(variant):.4f}")

        # Risk metrics
        lines.append("")
        lines.append("  RISK METRICS")
        lines.append(f"  Max Drawdown:   ${self.max_drawdown(variant):,.2f}")
        lines.append(f"  Avg Drawdown:   ${self.avg_drawdown(variant):,.2f}")
        lines.append(f"  Sharpe Ratio:   {self.sharpe_ratio(variant):.2f}")
        lines.append(f"  Sortino Ratio:  {self.sortino_ratio(variant):.2f}")
        lines.append(f"  Calmar Ratio:   {self.calmar_ratio(variant):.2f}")

        # Breakdowns
        for label, breakdown in [
            ("BY CITY", self.breakdown_by_city(variant)),
            ("BY HORIZON", self.breakdown_by_horizon(variant)),
            ("BY REGIME", self.breakdown_by_regime(variant)),
            ("BY SIDE", self.breakdown_by_side(variant)),
            ("BY SOURCE", self.breakdown_by_source(variant)),
        ]:
            lines.append("")
            lines.append(f"  {label}")
            lines.append(f"  {'Key':<16} {'WR':>6} {'PnL':>10} {'Trades':>7} {'AvgEV':>8}")
            lines.append(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*7} {'-'*8}")
            for key in sorted(breakdown.keys(), key=str):
                s = breakdown[key]
                lines.append(
                    f"  {str(key):<16} {s['win_rate']:>5.1%} {s['pnl']:>10.2f} {s['trades']:>7} {s['avg_ev']:>8.4f}"
                )

        # Fragility
        lines.append("")
        lines.append("  FRAGILITY NOTES")
        for note in self.fragility_notes(variant):
            lines.append(f"  {note}")

        # Robustness score
        score, penalties = self.robustness_score(variant)
        verdict = self.robustness_verdict(score)
        lines.append("")
        lines.append("  ROBUSTNESS ASSESSMENT")
        lines.append(f"  Score:    {score:.0f} / 100")
        lines.append(f"  Verdict:  {verdict}")
        if penalties:
            lines.append("")
            lines.append("  Penalties:")
            for p in penalties:
                lines.append(f"    - {p}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    # -- CSV export ---------------------------------------------------------

    def export_csv(self, filepath: str) -> None:
        """Write all trades to CSV."""
        if not self.trades:
            return
        field_names = [f.name for f in fields(BacktestTrade)]
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for t in self.trades:
                row = asdict(t)
                # date -> ISO string for CSV
                row["target_date"] = t.target_date.isoformat()
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Sensitivity analyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """Parameter sweep definitions and result formatting."""

    SWEEPS: Dict[str, List[Any]] = {
        "kelly_fraction": [0.05, 0.10, 0.15, 0.20, 0.25],
        "min_edge": [0.04, 0.06, 0.08, 0.10, 0.12],
        "base_sigma": [1.0, 1.5, 2.0, 2.5, 3.0],
        "confidence_floor": [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_book_depth": [2.0, 5.0, 10.0, 20.0],
    }

    @staticmethod
    def render_sweep_result(
        param: str,
        results: Dict[Any, Dict[str, float]],
        current_value: Any,
    ) -> str:
        """Formatted output for a single parameter sweep.

        *results* maps param_value -> {pnl, sharpe, win_rate, trades}.
        *current_value* is the production setting (marked with *).
        """
        lines: list[str] = []
        lines.append(f"  SWEEP: {param}")
        lines.append(f"  {'Value':>10} {'PnL':>10} {'Sharpe':>8} {'WR':>7} {'Trades':>7}")
        lines.append(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
        for val in sorted(results.keys()):
            r = results[val]
            marker = " *" if val == current_value else "  "
            lines.append(
                f"  {str(val):>10}{marker} {r.get('pnl', 0):>10.2f} "
                f"{r.get('sharpe', 0):>8.2f} {r.get('win_rate', 0):>6.1%} {r.get('trades', 0):>7.0f}"
            )
        return "\n".join(lines)
