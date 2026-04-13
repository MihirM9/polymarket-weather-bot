"""
decision_engine.py — Module 3: Decision & Risk Engine
======================================================
Ref: Research §5 (Quantifying the edge and sizing trades),
     §5.2 (Example trade math: EV_No = (1-p_true)*(1-c) - p_true*(1+c)),
     §5.3 (Kelly sizing for 100-1000 bankrolls: f* = ((1-q) - p_true)/(1-q), use 0.1-0.25 fractional),
     §5.4 (Daily loss caps and multi-market scaling: 6 cities × 10+ buckets, $5-30 per).

For each matched (market, forecast) pair:
  1. Compute forecast probability per bucket.
  2. Evaluate EV for both Yes and No sides of each outcome.
  3. Apply Kelly criterion with fractional sizing.
  4. Enforce per-market and daily caps.
  5. Output ranked trade signals.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from config import Config, cfg
from forecasting import CityForecast, bucket_probabilities

from .markets import TemperatureMarket

logger = logging.getLogger(__name__)

CORRELATION_GROUPS = {
    "northeast": ["New York", "Chicago"],
    "gulf": ["Houston", "Miami"],
    "south_central": ["Dallas", "Houston"],
    "west": ["Los Angeles"],
}
CORRELATED_GROUP_CAP_MULT = 1.5

@dataclass
class TradeSignal:
    """A ranked buy/sell decision ready for execution."""
    market_id: str
    city: str
    market_date: date
    outcome_label: str
    token_id: str
    side: str              # "BUY" (buy Yes shares) or "SELL" (buy No / sell Yes shares)
    p_true: float          # forecast probability for this bucket
    market_price: float    # current Yes price
    ev: float              # expected value per $1 risked
    edge: float            # p_true - market_price (or inverted for No)
    kelly_fraction: float  # raw Kelly fraction
    position_size_usd: float  # sized in USDC
    price_limit: float     # limit order price
    rationale: str


def _ev_yes(p_true: float, price: float, fee: float) -> float:
    """
    EV of buying Yes at `price`.
    Payoff: win → (1 - price) * (1 - fee);  lose → -price.
    EV = p_true * (1 - price) * (1 - fee) - (1 - p_true) * price
    """
    return p_true * (1.0 - price) * (1.0 - fee) - (1.0 - p_true) * price


def _ev_no(p_true: float, price_yes: float, fee: float) -> float:
    """
    EV of buying No at price_no = 1 - price_yes.
    Ref: §5.2 — EV_No = (1-p_true)*(1-c) - p_true*(1+c) adjusted for pricing.
    Payoff: win → price_yes * (1 - fee);  lose → -(1 - price_yes).
    EV = (1-p_true) * price_yes * (1-fee) - p_true * (1 - price_yes)
    """
    price_no = 1.0 - price_yes
    return (1.0 - p_true) * price_yes * (1.0 - fee) - p_true * price_no


def _kelly_yes(p_true: float, price: float) -> float:
    """
    Kelly fraction for buying Yes.
    b = (1 - price) / price  (odds)
    f* = (b*p - q) / b  where q = 1 - p_true
    """
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - p_true
    f = (b * p_true - q) / b
    return max(0.0, f)


def _kelly_no(p_true: float, price_yes: float) -> float:
    """
    Kelly fraction for buying No.
    Ref: §5.3 — f* = ((1-q) - p_true)/(1-q) for No side.
    b = price_yes / (1 - price_yes)
    f* = (b*(1-p_true) - p_true) / b
    """
    price_no = 1.0 - price_yes
    if price_no <= 0 or price_no >= 1:
        return 0.0
    b = price_yes / price_no
    p_no = 1.0 - p_true
    f = (b * p_no - p_true) / b
    return max(0.0, f)


class DecisionEngine:
    """
    Evaluates all matched market-forecast pairs and produces sized trade signals.
    Ref: §6.2 (Research/modeling → probs → EV), §5.4 (multi-market scaling).

    Upgrades over naive implementation:
      - Confidence-adaptive Kelly: temper MORE when forecast is uncertain,
        temper LESS when forecast is high-confidence. This means the bot
        bets bigger only when it has a strong informational edge.
      - Dynamic edge threshold: require a larger edge when σ is wide or
        weather regime is volatile. This filters out "looks profitable but
        the model is uncertain" situations that cause variance blowups.
    """

    def __init__(self, config: Config = cfg):
        self.config = config
        self.daily_pnl: float = 0.0         # tracked across the day
        self.daily_exposure: float = 0.0     # total $ at risk today
        self._today: Optional[date] = None
        self._group_exposure: Dict[str, float] = {}

    def _reset_daily_if_needed(self):
        today = date.today()
        if self._today != today:
            self.daily_pnl = 0.0
            self.daily_exposure = 0.0
            self._group_exposure = {}
            self._today = today

    def update_pnl(self, realized_pnl: float):
        """Called by execution module when trades settle."""
        self._reset_daily_if_needed()
        self.daily_pnl += realized_pnl

    def is_shutdown(self) -> bool:
        """Check daily loss cap (§5.4: -$50 shutdown for $500 bankroll)."""
        self._reset_daily_if_needed()
        if self.daily_pnl < -self.config.daily_loss_cap:
            logger.warning(
                f"DAILY LOSS CAP BREACHED: PnL={self.daily_pnl:.2f}, cap=-{self.config.daily_loss_cap}"
            )
            return True
        return False

    def _adaptive_kelly_fraction(self, base_fraction: float, confidence: float) -> float:
        """
        Scale Kelly fraction by forecast confidence.

        Instead of a flat multiplier (e.g., always 0.15x Kelly), we scale:
          - confidence=1.0 → use base_fraction * 1.5 (up to 22.5% Kelly at 0.15 base)
          - confidence=0.5 → use base_fraction * 0.75
          - confidence=0.0 → use base_fraction * 0.25 (minimal sizing)

        The formula: tempered = base * (0.25 + 1.25 * confidence)
        This means high-confidence forecasts get up to 1.5x the base Kelly,
        while low-confidence ones are damped to 0.25x.
        """
        scaling = 0.25 + 1.25 * confidence  # range: [0.25, 1.50]
        scaling = min(scaling, self.config.max_kelly_mult)
        return base_fraction * scaling

    @staticmethod
    def _dynamic_edge_threshold(base_edge: float, confidence: float, sigma: float) -> float:
        """
        Require larger edge when forecast is uncertain.

        Base edge threshold (e.g., 0.08) is the minimum for a perfect forecast.
        As confidence drops or σ widens, we demand more edge to compensate:

          threshold = base_edge + uncertainty_penalty

        where uncertainty_penalty scales with (1 - confidence) and σ.
        This filters out trades where EV looks good on paper but the model
        is too uncertain to trust — reducing variance without killing good trades.

        Examples at base_edge=0.08:
          confidence=0.95, σ=1.5 → threshold ≈ 0.083 (barely changed — go for it)
          confidence=0.70, σ=3.0 → threshold ≈ 0.115 (need ~12% edge)
          confidence=0.40, σ=5.0 → threshold ≈ 0.170 (need ~17% edge — very selective)
        """
        # Uncertainty penalty: up to +0.15 edge required at worst case
        uncertainty_penalty = (1.0 - confidence) * 0.05 + max(0, sigma - 1.5) * 0.02
        return base_edge + uncertainty_penalty

    @staticmethod
    def _time_decay_ev_threshold(base_threshold: float, days_to_resolution: int) -> float:
        """Scale EV threshold by sqrt of days to resolution — farther out markets need higher EV."""
        return base_threshold * math.sqrt(max(1, days_to_resolution))

    def _get_city_groups(self, city: str) -> List[str]:
        """Return all correlation group names that contain this city."""
        return [group for group, cities in CORRELATION_GROUPS.items() if city in cities]

    def _check_group_exposure(self, city: str, proposed_size: float) -> float:
        """Cap proposed_size so no correlation group exceeds its cap."""
        group_cap = self.config.per_market_max_pct * self.config.bankroll * CORRELATED_GROUP_CAP_MULT
        for group in self._get_city_groups(city):
            current = self._group_exposure.get(group, 0.0)
            remaining = max(0.0, group_cap - current)
            proposed_size = min(proposed_size, remaining)
        return proposed_size

    def evaluate(
        self,
        matches: List[Tuple[TemperatureMarket, CityForecast]],
        tracker=None,
    ) -> List[TradeSignal]:
        """
        For each matched market-forecast pair:
          1. Compute bucket probabilities from forecast.
          2. Evaluate EV for Yes and No on each outcome.
          3. Size with confidence-adaptive Kelly.
          4. Filter by dynamic edge thresholds.
          5. Return ranked signals.

        v3: accepts optional PositionTracker for accurate exposure-based sizing.
        """
        self._reset_daily_if_needed()

        if self.is_shutdown():
            logger.warning("Bot is in shutdown mode — no new trades")
            return []

        signals: List[TradeSignal] = []

        for mkt, fc in matches:
            # Skip unstable forecasts (§ adaptive: skip volatile/hurricane/rapid changes)
            if not fc.is_stable:
                logger.info(f"Skipping unstable forecast: {fc.city} {fc.forecast_date}")
                continue

            # Build bucket list from market outcomes
            buckets = []
            for outcome in mkt.outcomes:
                buckets.append((outcome.bucket_low, outcome.bucket_high))

            if not buckets:
                continue

            # Compute forecast probabilities per bucket (§5.1)
            probs = bucket_probabilities(fc.high_f, fc.sigma, buckets)

            # Compute adaptive parameters for this forecast
            adaptive_kelly = self._adaptive_kelly_fraction(self.config.kelly_fraction, fc.confidence)
            dynamic_edge = self._dynamic_edge_threshold(self.config.min_edge, fc.confidence, fc.sigma)

            days_to_res = max(1, (mkt.market_date - date.today()).days)
            time_adj_ev_threshold = self._time_decay_ev_threshold(self.config.min_ev_threshold, days_to_res)

            for i, outcome in enumerate(mkt.outcomes):
                p_true = probs.get(i, 0.0)
                price_yes = outcome.price_yes

                if price_yes <= 0.01 or price_yes >= 0.99:
                    continue  # illiquid / degenerate

                # Skip outcomes on cancel-replace cooldown (v3.1)
                if tracker is not None and tracker.is_cooled_down(mkt.market_id, outcome.outcome_label):
                    logger.debug(f"Skipping cooled-down outcome: {outcome.outcome_label}")
                    continue

                # Skip outcomes that already have an active order (deduplication)
                if tracker is not None and tracker.has_active_order(mkt.market_id, outcome.outcome_label):
                    logger.debug(f"Skipping duplicate: active order exists for {outcome.outcome_label}")
                    continue

                # --- Evaluate BUY YES side ---
                ev_y = _ev_yes(p_true, price_yes, fee=self.config.maker_fee_rate)
                edge_y = p_true - price_yes
                kelly_y = _kelly_yes(p_true, price_yes) * adaptive_kelly

                if ev_y > time_adj_ev_threshold and edge_y > dynamic_edge:
                    size = self._size_position(kelly_y, tracker, city=mkt.city)
                    if size > 0:
                        signals.append(TradeSignal(
                            market_id=mkt.market_id,
                            city=mkt.city,
                            market_date=mkt.market_date,
                            outcome_label=outcome.outcome_label,
                            token_id=outcome.token_id,
                            side="BUY",
                            p_true=p_true,
                            market_price=price_yes,
                            ev=ev_y,
                            edge=edge_y,
                            kelly_fraction=kelly_y,
                            position_size_usd=size,
                            price_limit=min(price_yes + 0.02, p_true - 0.02),  # don't overpay
                            rationale=(
                                f"BUY YES: p_true={p_true:.3f} vs mkt={price_yes:.3f}, "
                                f"edge={edge_y:.3f} (thresh={dynamic_edge:.3f}), EV={ev_y:.3f}, "
                                f"forecast={fc.high_f:.0f}°F±{fc.sigma:.1f} ({fc.weather_regime}), "
                                f"conf={fc.confidence:.2f}, kelly_adj={adaptive_kelly:.3f}, "
                                f"bucket=[{outcome.bucket_low}–{outcome.bucket_high}], days_out={days_to_res}"
                            ),
                        ))

                # --- Evaluate BUY NO side (sell Yes / buy No) ---
                # This is the "sell extremes" play (§2.3: selling extremes No at 0.93-0.99)
                ev_n = _ev_no(p_true, price_yes, fee=self.config.maker_fee_rate)
                edge_n = (1.0 - p_true) - (1.0 - price_yes)  # = price_yes - p_true
                kelly_n = _kelly_no(p_true, price_yes) * adaptive_kelly

                if ev_n > time_adj_ev_threshold and edge_n > dynamic_edge:
                    size = self._size_position(kelly_n, tracker, city=mkt.city)
                    if size > 0:
                        # SELL = sell YES tokens. Limit price is the YES price
                        # we're willing to sell at (near market), NOT the NO price.
                        # Sell above our estimate of true value, near current market.
                        sell_limit = max(price_yes - 0.02, p_true + 0.02)
                        # Sanity: never sell YES above 50¢ when betting NO
                        if sell_limit > 0.50:
                            logger.debug(
                                f"SELL limit {sell_limit:.3f} > 0.50 for {outcome.outcome_label}, skipping"
                            )
                            continue
                        signals.append(TradeSignal(
                            market_id=mkt.market_id,
                            city=mkt.city,
                            market_date=mkt.market_date,
                            outcome_label=outcome.outcome_label,
                            token_id=outcome.token_id,
                            side="SELL",
                            p_true=p_true,
                            market_price=price_yes,
                            ev=ev_n,
                            edge=edge_n,
                            kelly_fraction=kelly_n,
                            position_size_usd=size,
                            price_limit=sell_limit,
                            rationale=(
                                f"BUY NO (sell extreme): p_true={p_true:.3f}, mkt_yes={price_yes:.3f}, "
                                f"edge_no={edge_n:.3f} (thresh={dynamic_edge:.3f}), EV={ev_n:.3f}, "
                                f"forecast={fc.high_f:.0f}°F±{fc.sigma:.1f} ({fc.weather_regime}), "
                                f"conf={fc.confidence:.2f}, kelly_adj={adaptive_kelly:.3f}, "
                                f"bucket=[{outcome.bucket_low}–{outcome.bucket_high}], days_out={days_to_res}"
                            ),
                        ))

        # Sort by EV descending (best trades first)
        signals.sort(key=lambda s: s.ev, reverse=True)
        logger.info(f"Decision engine produced {len(signals)} trade signals")
        return signals

    def _size_position(self, kelly_frac: float, tracker=None, city: str = "") -> float:
        """
        Convert Kelly fraction to USDC size with caps.
        v3: Uses PositionTracker's realized_exposure when available,
        falling back to self.daily_exposure for backward compat / dry-run.
        v4: Correlated exposure group caps.
        """
        raw_size = kelly_frac * self.config.bankroll

        # Per-market cap
        market_cap = self.config.per_market_max_pct * self.config.bankroll
        raw_size = min(raw_size, market_cap)

        # Absolute cap
        raw_size = min(raw_size, self.config.max_position_usd)

        # Check daily exposure — use tracker if available (ground truth),
        # otherwise fall back to internal counter
        if tracker is not None:
            current_exposure = tracker.total_exposure
        else:
            current_exposure = self.daily_exposure

        remaining_budget = self.config.bankroll * 0.3 - current_exposure
        raw_size = min(raw_size, max(0.0, remaining_budget))

        # Correlated exposure group cap (v4)
        if city:
            raw_size = self._check_group_exposure(city, raw_size)

        # Floor: don't bother with dust
        if raw_size < 1.0:
            return 0.0

        self.daily_exposure += raw_size

        # Update group exposure tracking
        if city:
            for group in self._get_city_groups(city):
                self._group_exposure[group] = self._group_exposure.get(group, 0.0) + raw_size

        return round(raw_size, 2)
