# tests/test_backtest_pricing.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

def test_tail_buckets_overpriced():
    """Tail buckets should have prices higher than true probabilities."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.02, 0.05, 0.15, 0.30, 0.25, 0.13, 0.07, 0.03]

    random.seed(42)
    results = {"tail_bias": [], "mode_bias": []}
    for _ in range(500):
        prices = model.generate_prices(true_probs, days_out=3)
        results["tail_bias"].append(prices[0] - true_probs[0])
        results["tail_bias"].append(prices[-1] - true_probs[-1])
        results["mode_bias"].append(prices[3] - true_probs[3])

    avg_tail_bias = sum(results["tail_bias"]) / len(results["tail_bias"])
    avg_mode_bias = sum(results["mode_bias"]) / len(results["mode_bias"])

    assert avg_tail_bias > 0.02, f"Tail bias {avg_tail_bias:.3f} too low"
    assert avg_mode_bias < 0.0, f"Mode bias {avg_mode_bias:.3f} should be negative"

def test_prices_clamped():
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.001, 0.999]

    random.seed(42)
    for _ in range(100):
        prices = model.generate_prices(true_probs, days_out=5)
        for p in prices:
            assert 0.02 <= p <= 0.98, f"Price {p} out of bounds"

def test_convergence_near_resolution():
    """Prices should be more accurate (less biased) at days_out=0 vs days_out=5."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]

    random.seed(42)
    errors_far = []
    errors_near = []
    for _ in range(500):
        prices_5 = model.generate_prices(true_probs, days_out=5)
        prices_0 = model.generate_prices(true_probs, days_out=0)
        errors_far.append(sum(abs(p - t) for p, t in zip(prices_5, true_probs)))
        errors_near.append(sum(abs(p - t) for p, t in zip(prices_0, true_probs)))

    avg_far = sum(errors_far) / len(errors_far)
    avg_near = sum(errors_near) / len(errors_near)
    assert avg_near < avg_far, f"Near-resolution error ({avg_near:.3f}) should be < far ({avg_far:.3f})"

def test_calibrate_from_markets():
    """Test that calibration updates model parameters."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    closed_data = [
        (0.0, 0.15, 0.0),
        (0.0, 0.12, 0.0),
        (0.5, 0.25, 1.0),
        (0.5, 0.28, 1.0),
        (1.0, 0.10, 0.0),
        (1.0, 0.14, 0.0),
    ]
    model.calibrate(closed_data)
    assert model.tail_overpricing > 0.0
