"""
amm_config.py -- Market-Making Bot Configuration
==================================================
Extends the base config with MM-specific parameters for spread quoting,
inventory management, event detection, and liquidity reward targeting.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

from config import cfg as base_cfg


@dataclass
class AMMConfig:
    """Market-making specific configuration layered on top of base Config."""

    # --- Quoting parameters ---
    # Half-spread: how far from mid to place quotes (in price units, 0-1 scale)
    # 1-3 cents inside mid as specified in the strategy doc
    half_spread: float = float(os.getenv("AMM_HALF_SPREAD", "0.02"))
    min_spread: float = float(os.getenv("AMM_MIN_SPREAD", "0.01"))
    max_spread: float = float(os.getenv("AMM_MAX_SPREAD", "0.05"))

    # Spread tightening: how aggressively to improve on best bid/ask
    # 0.005 = place 0.5 cent inside current best
    spread_improvement: float = float(os.getenv("AMM_SPREAD_IMPROVEMENT", "0.005"))

    # --- Cycle timing ---
    # MM cycles run faster than directional bot (30-60s vs 120s)
    cycle_interval_sec: int = int(os.getenv("AMM_CYCLE_SEC", "45"))

    # --- Inventory / position limits ---
    # Max notional per side per market (USD)
    max_position_per_market: float = float(os.getenv("AMM_MAX_POS_MARKET", "50.0"))
    # Max total inventory across all markets (USD)
    max_total_inventory: float = float(os.getenv("AMM_MAX_TOTAL_INV", "500.0"))
    # Inventory skew threshold: when net position exceeds this fraction of
    # max_position_per_market, shift quotes to reduce exposure
    inventory_skew_threshold: float = float(os.getenv("AMM_SKEW_THRESHOLD", "0.5"))
    # How much to shift mid when inventory is skewed (cents per unit of skew)
    inventory_skew_shift: float = float(os.getenv("AMM_SKEW_SHIFT", "0.01"))

    # --- Order sizing ---
    # Base quote size (USD) per side
    base_quote_size: float = float(os.getenv("AMM_QUOTE_SIZE", "5.0"))
    # Min quote size to bother placing
    min_quote_size: float = float(os.getenv("AMM_MIN_QUOTE_SIZE", "1.0"))

    # --- Risk controls ---
    # Max adverse selection rate before pulling quotes (fraction)
    max_adverse_selection: float = float(os.getenv("AMM_MAX_ADV_SEL", "0.25"))
    # Daily loss cap for MM operations (USD)
    mm_daily_loss_cap: float = float(os.getenv("AMM_DAILY_LOSS_CAP", "25.0"))
    # Max spread capture loss per trade (USD) -- pull quotes if exceeded
    max_loss_per_trade: float = float(os.getenv("AMM_MAX_LOSS_TRADE", "2.0"))

    # --- Stale quote management ---
    # Cancel and refresh quotes after this many seconds
    quote_refresh_sec: int = int(os.getenv("AMM_QUOTE_REFRESH_SEC", "120"))
    # Cooldown after cancel before re-quoting same market (seconds)
    requote_cooldown_sec: int = int(os.getenv("AMM_REQUOTE_COOLDOWN_SEC", "30"))

    # --- Market selection filters ---
    # Minimum 24h volume to consider a market for MM (USD)
    min_market_volume: float = float(os.getenv("AMM_MIN_VOLUME", "5000.0"))
    # Maximum 24h volume -- avoid hyper-competitive markets
    max_market_volume: float = float(os.getenv("AMM_MAX_VOLUME", "200000.0"))
    # Minimum liquidity (USD) -- need existing book to quote against
    min_market_liquidity: float = float(os.getenv("AMM_MIN_LIQUIDITY", "500.0"))
    # Maximum spread to enter (wider = more profit but more risk)
    max_entry_spread: float = float(os.getenv("AMM_MAX_ENTRY_SPREAD", "0.15"))
    # Minimum spread to enter (too tight = no profit after fees)
    min_entry_spread: float = float(os.getenv("AMM_MIN_ENTRY_SPREAD", "0.02"))
    # Max markets to quote simultaneously
    max_active_markets: int = int(os.getenv("AMM_MAX_MARKETS", "10"))

    # --- Event detection (pull quotes before news) ---
    # Pull quotes N seconds before known event times
    event_pullback_sec: int = int(os.getenv("AMM_EVENT_PULLBACK_SEC", "300"))
    # Widen spread by this factor during high-volatility periods
    volatility_spread_mult: float = float(os.getenv("AMM_VOL_SPREAD_MULT", "2.0"))

    # --- Fee optimization ---
    # Maker fee rate (0 on Polymarket for limit orders)
    maker_fee: float = float(os.getenv("AMM_MAKER_FEE", "0.0"))
    # Taker fee rate (for when we need to exit positions aggressively)
    taker_fee: float = float(os.getenv("AMM_TAKER_FEE", "0.02"))

    # --- Market categories to target ---
    # Keywords for discovering MM-suitable markets on Gamma API
    target_categories: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv(
            "AMM_CATEGORIES",
            "temperature,weather,sports,esports,entertainment"
        ).split(",")
    ])

    # --- Liquidity reward tracking ---
    # Whether to optimize for liquidity reward programs
    track_rewards: bool = os.getenv("AMM_TRACK_REWARDS", "true").lower() == "true"
    # Min uptime fraction to qualify for rewards (if applicable)
    min_uptime_for_rewards: float = float(os.getenv("AMM_MIN_UPTIME", "0.80"))
    # Max spread to qualify for reward programs
    reward_max_spread: float = float(os.getenv("AMM_REWARD_MAX_SPREAD", "0.05"))
    # Min size to qualify for reward programs (USD equivalent)
    reward_min_size: float = float(os.getenv("AMM_REWARD_MIN_SIZE", "5.0"))

    # --- Inherit base config ---
    @property
    def bankroll(self) -> float:
        return base_cfg.bankroll

    @property
    def is_live(self) -> bool:
        return base_cfg.is_live

    @property
    def private_key(self) -> str:
        return base_cfg.private_key

    @property
    def polymarket_host(self) -> str:
        return base_cfg.polymarket_host

    @property
    def chain_id(self) -> int:
        return base_cfg.chain_id

    @property
    def funder(self) -> str:
        return base_cfg.funder

    @property
    def telegram_token(self) -> str:
        return base_cfg.telegram_token

    @property
    def telegram_chat_id(self) -> str:
        return base_cfg.telegram_chat_id


# Singleton
amm_cfg = AMMConfig()
