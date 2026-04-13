import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
strategies = pytest.importorskip("hypothesis.strategies")

from decision_engine import _ev_no, _ev_yes, _kelly_no, _kelly_yes
from forecast_scanner import bucket_probabilities

given = hypothesis.given
settings = hypothesis.settings


@settings(max_examples=100)
@given(
    mu=strategies.floats(min_value=-20, max_value=120, allow_nan=False, allow_infinity=False),
    sigma=strategies.floats(min_value=0.2, max_value=15, allow_nan=False, allow_infinity=False),
)
def test_bucket_probabilities_sum_to_one_for_exhaustive_partition(mu, sigma):
    buckets = [
        (None, 0.0),
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
        (60.0, 80.0),
        (80.0, 100.0),
        (100.0, None),
    ]
    probs = bucket_probabilities(mu, sigma, buckets)

    total = sum(probs.values())
    assert 0.999 <= total <= 1.001
    assert all(0.0 <= value <= 1.0 for value in probs.values())


@settings(max_examples=100)
@given(
    p_true=strategies.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    price=strategies.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_kelly_never_negative(p_true, price):
    assert _kelly_yes(p_true, price) >= 0.0
    assert _kelly_no(p_true, price) >= 0.0


@settings(max_examples=100)
@given(
    price=strategies.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    fee=strategies.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_ev_is_fair_near_break_even(price, fee):
    ev_yes = _ev_yes(price, price, fee)
    ev_no = _ev_no(price, price, fee)

    assert ev_yes <= 1e-9
    assert ev_no <= 1e-9


@settings(max_examples=100)
@given(
    price=strategies.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    fee=strategies.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    p1=strategies.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False),
    p2=strategies.floats(min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_ev_yes_monotonic_in_true_probability(price, fee, p1, p2):
    assert p1 < p2
    assert _ev_yes(p1, price, fee) <= _ev_yes(p2, price, fee)
    assert _ev_no(p1, price, fee) >= _ev_no(p2, price, fee)
