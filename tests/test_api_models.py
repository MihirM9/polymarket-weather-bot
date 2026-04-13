import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.models import GammaMarketWire, NWSPointsResponse, validate_model


def test_gamma_market_wire_parses_stringified_lists():
    payload = {
        "id": "123",
        "question": "What will the highest temperature be in Miami on July 15?",
        "outcomePrices": "[0.35, 0.65]",
        "clobTokenIds": "[\"token_yes\", \"token_no\"]",
        "volume": "1234.5",
        "liquidity": "250.0",
    }

    parsed = GammaMarketWire.model_validate(payload)

    assert parsed.outcomePrices == [0.35, 0.65]
    assert parsed.clobTokenIds == ["token_yes", "token_no"]
    assert parsed.volume == 1234.5


def test_validate_model_returns_none_on_invalid_payload():
    payload = {"properties": {"gridId": "MFL", "gridX": "oops"}}

    parsed = validate_model(payload, NWSPointsResponse, label="test-nws-points")

    assert parsed is None
