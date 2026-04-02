# backtest_pricing.py
"""
backtest_pricing.py — Calibrated Mispricing Model for Backtesting
==================================================================
Generates synthetic market prices that reproduce the systematic biases
observed in real Polymarket temperature markets:
  - Tail buckets are overpriced (retail loves long shots)
  - Mode bucket is underpriced (retail underweights the most likely outcome)
  - Prices converge toward truth as resolution approaches

Calibrated from real Gamma API closed market data.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MispricingModel:
    """
    Generate synthetic market prices with calibrated retail mispricing.

    Includes bid-ask spread and slippage modeling for realistic execution.
    """

    tail_overpricing: float = 0.07
    mode_underpricing: float = -0.04
    noise_sigma: float = 0.03
    convergence_rate: float = 1.0
    half_spread: float = 0.015      # half the bid-ask spread (3¢ total)
    slippage_sigma: float = 0.005   # random slippage on fill price

    def generate_prices(
        self,
        true_probs: List[float],
        days_out: int,
    ) -> List[float]:
        n = len(true_probs)
        if n == 0:
            return []

        mode_idx = max(range(n), key=lambda i: true_probs[i])

        # bias_scale: 1.0 when far from resolution, 0.0 at resolution
        bias_scale = min(1.0, (days_out / 7.0) * self.convergence_rate)

        prices = []
        for i, p_true in enumerate(true_probs):
            if n > 1:
                tail_distance = abs(i - mode_idx) / max(1, (n - 1) / 2)
                tail_distance = min(1.0, tail_distance)
            else:
                tail_distance = 0.0

            tail_bias = self.tail_overpricing * tail_distance
            mode_bias = self.mode_underpricing if i == mode_idx else 0.0
            total_bias = (tail_bias + mode_bias) * bias_scale
            noise = random.gauss(0, self.noise_sigma * bias_scale)

            raw_price = p_true + total_bias + noise
            prices.append(max(0.02, min(0.98, raw_price)))

        return prices

    def apply_execution_cost(self, price: float, side: str) -> float:
        """
        Apply bid-ask spread and slippage to simulate realistic execution.

        BUY YES: you pay the ask (mid + half_spread + slippage)
        SELL/BUY NO: you pay 1 - bid, so effective YES price drops
        """
        slippage = abs(random.gauss(0, self.slippage_sigma))
        if side == "BUY":
            # Buying YES: worse fill = higher price
            fill_price = price + self.half_spread + slippage
        else:
            # Selling YES / Buying NO: worse fill = lower YES price
            fill_price = price - self.half_spread - slippage
        return max(0.02, min(0.98, fill_price))

    def calibrate(
        self,
        closed_data: List[Tuple[float, float, float]],
    ):
        if not closed_data:
            return

        tail_biases = []
        mode_biases = []
        all_noise = []

        for pos, price, outcome in closed_data:
            bias = price - outcome
            if pos < 0.2 or pos > 0.8:
                tail_biases.append(bias)
            elif 0.4 <= pos <= 0.6:
                mode_biases.append(bias)
            all_noise.append(abs(bias))

        if tail_biases:
            self.tail_overpricing = max(0.01, sum(tail_biases) / len(tail_biases))
        if mode_biases:
            self.mode_underpricing = min(-0.01, sum(mode_biases) / len(mode_biases))
        if all_noise:
            self.noise_sigma = max(0.01, sum(all_noise) / len(all_noise) * 0.5)
