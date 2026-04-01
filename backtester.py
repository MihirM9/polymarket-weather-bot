# backtester.py
"""
backtester.py — Historical Backtester for Weather Trading Bot
==============================================================
Replays 12-18 months of historical data through the exact same
DecisionEngine.evaluate() pipeline the live bot uses.

Produces a scorecard with Sharpe, win rate, drawdown, breakdowns,
parameter sensitivity analysis, and fragility notes.

Usage:
    python backtester.py                         # full backtest
    python backtester.py --quick                 # realistic only, no sensitivity
    python backtester.py --sensitivity           # include parameter sweeps
    python backtester.py --fetch-only            # just populate data caches
    python backtester.py --start 2025-01-01 --end 2026-03-31
    python backtester.py --cities "Miami,New York"
"""

import argparse
import asyncio
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from backtest_data import HistoricalDataLoader
from backtest_forecast import HistoricalForecastApproximator, SyntheticForecast
from backtest_pricing import MispricingModel
from backtest_scorecard import BacktestScorecard, BacktestTrade, SensitivityAnalyzer
from backtest_tracker import MockTracker
from config import cfg
from decision_engine import DecisionEngine, TradeSignal
from forecast_scanner import CityForecast, bucket_probabilities
from polymarket_parser import TemperatureMarket, MarketOutcome, _parse_bucket
from resolution_tracker import ResolutionTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtester")


# Standard temperature buckets for synthetic markets (degrees F)
STANDARD_BUCKETS = [
    (None, 30), (30, 35), (35, 40), (40, 45), (45, 50),
    (50, 55), (55, 60), (60, 65), (65, 70), (70, 75),
    (75, 80), (80, 85), (85, 90), (90, 95), (95, 100),
    (100, None),
]


@dataclass
class BacktestResult:
    """Container for all backtest output."""
    trades: List[BacktestTrade] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    methodology_notes: List[str] = field(default_factory=list)

    def record(self, trade: BacktestTrade):
        self.trades.append(trade)


class BacktestEngine:
    """
    Replays historical city/date pairs through DecisionEngine.evaluate().

    For each city/date:
    1. Generate synthetic forecasts (realistic + optimistic)
    2. Build or load market prices
    3. Run the exact same DecisionEngine.evaluate()
    4. Score signals against actual observed temperature
    """

    def __init__(self, bankroll: float = 500.0, seed: int = 42):
        self.bankroll = bankroll
        self.seed = seed
        self.loader = HistoricalDataLoader()
        self.approximator = HistoricalForecastApproximator()
        # Stronger mispricing defaults for backtesting — real Polymarket
        # markets show 8-15% tail overpricing and 5-8% mode underpricing.
        # Higher noise_sigma reflects real-world bid-ask spread and retail
        # mispricing variance. Overridden by calibrate() when real Gamma data
        # is available.
        self.pricing = MispricingModel(
            tail_overpricing=0.15,
            mode_underpricing=-0.12,
            noise_sigma=0.07,
        )

    def _build_market(
        self,
        city: str,
        target_date: date,
        days_out: int,
        forecast_high: float,
        forecast_sigma: float,
    ) -> TemperatureMarket:
        """
        Build a TemperatureMarket from real Gamma data or synthetic prices.
        """
        real_prices = self.loader.get_real_market_prices(city, target_date)

        if real_prices and len(real_prices) >= 3:
            outcomes = []
            for label, price_yes in real_prices.items():
                lo, hi = _parse_bucket(label)
                outcomes.append(MarketOutcome(
                    outcome_label=label,
                    token_id=f"backtest_{city}_{target_date}_{label}",
                    price_yes=price_yes,
                    price_no=round(1.0 - price_yes, 4),
                    bucket_low=lo,
                    bucket_high=hi,
                ))
        else:
            buckets = self._select_buckets(forecast_high)
            true_probs = bucket_probabilities(
                forecast_high, forecast_sigma, buckets
            )
            prices = self.pricing.generate_prices(
                [true_probs.get(i, 0.0) for i in range(len(buckets))],
                days_out=days_out,
            )

            outcomes = []
            for i, ((lo, hi), price) in enumerate(zip(buckets, prices)):
                label = self._bucket_label(lo, hi)
                outcomes.append(MarketOutcome(
                    outcome_label=label,
                    token_id=f"backtest_{city}_{target_date}_{label}",
                    price_yes=price,
                    price_no=round(1.0 - price, 4),
                    bucket_low=lo,
                    bucket_high=hi,
                ))

        outcomes.sort(key=lambda o: (o.bucket_low if o.bucket_low is not None else -999))

        return TemperatureMarket(
            market_id=f"backtest_{city}_{target_date.isoformat()}",
            question=f"What will the highest temperature be in {city} on {target_date}?",
            city=city,
            market_date=target_date,
            resolution_source="NWS",
            outcomes=outcomes,
            active=True,
        )

    def _select_buckets(
        self, forecast_high: float
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Select a subset of standard buckets centered on the forecast."""
        center_idx = 0
        for i, (lo, hi) in enumerate(STANDARD_BUCKETS):
            if lo is not None and hi is not None:
                if lo <= forecast_high < hi:
                    center_idx = i
                    break
            elif lo is None and hi is not None:
                if forecast_high < hi:
                    center_idx = i
                    break
            elif hi is None and lo is not None:
                if forecast_high >= lo:
                    center_idx = i
                    break

        start = max(0, center_idx - 4)
        end = min(len(STANDARD_BUCKETS), center_idx + 5)
        return STANDARD_BUCKETS[start:end]

    @staticmethod
    def _bucket_label(lo: Optional[float], hi: Optional[float]) -> str:
        if lo is None:
            return f"{hi:.0f}°F or below"
        if hi is None:
            return f"{lo:.0f}°F or higher"
        return f"{lo:.0f}-{hi - 1:.0f}°F"

    def _score_signal(
        self, signal: TradeSignal, actual_high: float
    ) -> Tuple[bool, float]:
        """Score a trade signal against the actual temperature."""
        lo, hi = _parse_bucket(signal.outcome_label)

        in_bucket = True
        if lo is not None and actual_high < lo:
            in_bucket = False
        if hi is not None and actual_high >= hi:
            in_bucket = False

        if signal.side == "BUY":
            won = in_bucket
        else:
            won = not in_bucket

        pnl = ResolutionTracker._calculate_pnl(
            signal.side, signal.price_limit,
            signal.position_size_usd, won,
            fee_rate=cfg.maker_fee_rate,
        )

        return won, pnl

    def run(
        self,
        cities: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Run the full backtest."""
        random.seed(self.seed)

        if cities is None:
            cities = cfg.cities
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today() - timedelta(days=1)

        result = BacktestResult(
            config_snapshot={
                "bankroll": self.bankroll,
                "seed": self.seed,
                "min_edge": cfg.min_edge,
                "kelly_fraction": cfg.kelly_fraction,
                "max_kelly_mult": cfg.max_kelly_mult,
                "fee_rate": cfg.fee_rate,
                "maker_fee_rate": cfg.maker_fee_rate,
            },
            methodology_notes=[
                "Forecasts approximated from NOAA climatology + calibrated bias + noise.",
                "Market prices from Gamma closed markets where available, synthetic elsewhere.",
                "No execution simulation (fills assumed at limit price).",
                f"Random seed: {self.seed}",
            ],
        )

        if not self.approximator._climatology:
            self.approximator.set_climatology(self.loader._climatology)

        engine = DecisionEngine()

        for city in cities:
            current = start_date
            while current <= end_date:
                actual_high = self.loader.get_actual_high(city, current)
                if actual_high is None:
                    current += timedelta(days=1)
                    continue

                # Use a separate MockTracker per city/date to enforce
                # deduplication within the day while allowing fresh
                # exposure budgets each day.
                mock_tracker = MockTracker(bankroll=self.bankroll)

                # Force DecisionEngine daily reset for this simulated day
                # by setting _today to a sentinel so _reset_daily_if_needed
                # fires, then overriding to current to prevent re-reset.
                engine._today = None
                engine._reset_daily_if_needed()

                for days_out in range(5, -1, -1):
                    forecast_r, forecast_o = self.approximator.generate(
                        city, current, days_out, actual_high
                    )

                    for variant_name, fc in [("realistic", forecast_r), ("optimistic", forecast_o)]:
                        market = self._build_market(
                            city, current, days_out,
                            fc.high_f, fc.sigma,
                        )

                        city_forecast = CityForecast(
                            city=city,
                            forecast_date=current,
                            high_f=fc.high_f,
                            sigma=fc.sigma,
                            confidence=fc.confidence,
                            weather_regime=fc.regime,
                            regime_multiplier=1.0,
                            is_stable=True,
                        )

                        signals = engine.evaluate(
                            [(market, city_forecast)],
                            tracker=mock_tracker,
                        )

                        for sig in signals:
                            won, pnl = self._score_signal(sig, actual_high)

                            mock_tracker.record_trade(
                                sig.market_id, sig.outcome_label,
                                sig.position_size_usd, city,
                            )

                            # Get bucket bounds for the trade
                            bucket_lo, bucket_hi = _parse_bucket(sig.outcome_label)

                            result.record(BacktestTrade(
                                city=city,
                                target_date=current,
                                days_out=days_out,
                                side=sig.side,
                                outcome_label=sig.outcome_label,
                                bucket_low=bucket_lo,
                                bucket_high=bucket_hi,
                                p_true=sig.p_true,
                                market_price=sig.market_price,
                                ev=sig.ev,
                                edge=sig.edge,
                                kelly_fraction=sig.kelly_fraction,
                                position_size_usd=sig.position_size_usd,
                                price_limit=sig.price_limit,
                                actual_high=actual_high,
                                won=won,
                                pnl=pnl,
                                variant=variant_name,
                                regime=fc.regime,
                            ))

                current += timedelta(days=1)

        logger.info(
            f"Backtest complete: {len(result.trades)} trades "
            f"({sum(1 for t in result.trades if t.variant == 'realistic')} realistic, "
            f"{sum(1 for t in result.trades if t.variant == 'optimistic')} optimistic)"
        )

        return result


# -- CLI --

async def fetch_data(loader: HistoricalDataLoader, cities: List[str], start: date, end: date):
    """Fetch all required historical data."""
    logger.info("Fetching NOAA daily highs...")
    await loader.fetch_daily_highs(cities, start, end)

    logger.info("Fetching Gamma closed markets...")
    await loader.fetch_gamma_closed_markets()

    logger.info("Building climatology from observations...")
    loader.load_climatology_from_actuals(loader._highs)


def main():
    parser = argparse.ArgumentParser(description="Backtest Weather Trading Strategy")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cities", type=str, default=None, help="Comma-separated cities")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sweeps")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data, don't run backtest")
    parser.add_argument("--quick", action="store_true", help="Realistic variant only, no sensitivity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Simulated bankroll")
    args = parser.parse_args()

    start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=365)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    cities = [c.strip() for c in args.cities.split(",")] if args.cities else cfg.cities

    engine = BacktestEngine(bankroll=args.bankroll, seed=args.seed)

    logger.info(f"Backtest: {start} -> {end}, cities: {cities}")
    asyncio.run(fetch_data(engine.loader, cities, start, end))

    if args.fetch_only:
        logger.info("Data fetch complete. Exiting.")
        return

    engine.approximator.set_climatology(engine.loader._climatology)

    if engine.loader._gamma_markets:
        logger.info("Calibrating mispricing model from Gamma closed markets...")
        cal_data = []
        for item in engine.loader._gamma_markets:
            city_name, mkt_date, lo, hi, price = engine.loader._parse_gamma_market(item)
            if city_name and lo is not None:
                pos = 0.5
                cal_data.append((pos, price, 0.0))
        if cal_data:
            engine.pricing.calibrate(cal_data)
            logger.info(
                f"Calibrated: tail_overpricing={engine.pricing.tail_overpricing:.3f}, "
                f"mode_underpricing={engine.pricing.mode_underpricing:.3f}"
            )

    logger.info("Running backtest...")
    result = engine.run(cities=cities, start_date=start, end_date=end)

    scorecard = BacktestScorecard(result.trades)
    print(scorecard.render("realistic"))

    if not args.quick:
        print(scorecard.render("optimistic"))

    csv_path = "data/backtest_results.csv"
    scorecard.export_csv(csv_path)
    logger.info(f"Results exported to {csv_path}")

    if args.sensitivity:
        logger.info("Running sensitivity sweeps...")
        analyzer = SensitivityAnalyzer()
        for param, values in analyzer.SWEEPS.items():
            sweep_results: Dict = {}
            for val in values:
                original = getattr(cfg, param, None)
                if original is not None:
                    setattr(cfg, param, val)
                    sweep_engine = BacktestEngine(bankroll=args.bankroll, seed=args.seed)
                    sweep_engine.loader = engine.loader
                    sweep_engine.approximator = engine.approximator
                    sweep_engine.pricing = engine.pricing
                    sweep_result = sweep_engine.run(cities=cities, start_date=start, end_date=end)
                    sweep_sc = BacktestScorecard(sweep_result.trades)
                    sweep_results[val] = {
                        "sharpe": sweep_sc.sharpe_ratio("realistic"),
                        "win_rate": sweep_sc.win_rate("realistic"),
                        "max_dd": sweep_sc.max_drawdown("realistic"),
                        "pnl": sweep_sc.total_pnl("realistic"),
                        "trades": sweep_sc.trade_count("realistic"),
                    }
                    setattr(cfg, param, original)

            current = getattr(cfg, param, 0)
            print(analyzer.render_sweep_result(param, sweep_results, current))
            print()

    sharpe = scorecard.sharpe_ratio("realistic")
    wr = scorecard.win_rate("realistic")
    dd = scorecard.max_drawdown("realistic")
    dd_pct = (dd / args.bankroll) * 100

    print("\n" + "=" * 62)
    print("  VERDICT")
    passed = True
    if sharpe < 1.5:
        print(f"  FAIL: Sharpe {sharpe:.2f} < 1.5")
        passed = False
    else:
        print(f"  PASS: Sharpe {sharpe:.2f} >= 1.5")
    if wr < 0.62:
        print(f"  FAIL: Win rate {wr:.1%} < 62%")
        passed = False
    else:
        print(f"  PASS: Win rate {wr:.1%} >= 62%")
    if dd_pct > 15:
        print(f"  FAIL: Max drawdown {dd_pct:.1f}% > 15%")
        passed = False
    else:
        print(f"  PASS: Max drawdown {dd_pct:.1f}% <= 15%")

    verdict = "STRATEGY VALIDATED" if passed else "STRATEGY NEEDS SURGERY"
    print(f"\n  {verdict}")
    print("=" * 62)


if __name__ == "__main__":
    main()
