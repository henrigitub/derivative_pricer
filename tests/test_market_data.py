"""Market-data parsing and validation using offline fixtures."""

from datetime import date, datetime
from unittest.mock import patch

import numpy as np
import pytest

from derivative_pricer.market_data import (
    compare_market,
    historical_volatility,
    parse_spx_history,
    parse_spx_snapshot,
    fetch_spx_snapshot,
    NEW_YORK,
)


def payload():
    return {
        "timestamp": "2026-09-25 20:30:00",
        "data": {
            "symbol": "^SPX",
            "current_price": 100,
            "last_trade_time": "2026-09-25T16:15:00",
            "options": [
                {"option": "SPXW261023C00100000", "bid": 2, "ask": 3},
                {"option": "SPX261023C00100000", "bid": 2, "ask": 3},
                {"option": "SPXW261023P00100000", "bid": 4, "ask": 3},
            ],
        },
    }


def test_parser_selects_pm_settled_quotes_and_rejects_crossed_markets():
    snapshot = parse_spx_snapshot(payload())
    assert len(snapshot.options) == 1
    assert snapshot.options[0].strike == 100
    assert snapshot.options[0].mid == 2.5
    assert snapshot.as_of == datetime(2026, 9, 25, 16, 15, tzinfo=NEW_YORK)
    # Replace the network response with a fixed payload.
    with patch("derivative_pricer.market_data._download_json", return_value=payload()):
        assert fetch_spx_snapshot().spot == 100


def test_historical_volatility_uses_log_returns_and_sample_std():
    returns = np.array([0.01, -0.02, 0.03])
    closes = 100 * np.exp(np.r_[0, np.cumsum(returns)])
    assert historical_volatility(closes) == pytest.approx(
        returns.std(ddof=1) * np.sqrt(252)
    )


@pytest.mark.parametrize("closes", [[100], [100, 0, 99], [100, float("nan"), 99]])
def test_bad_history_rejected(closes):
    with pytest.raises(ValueError):
        historical_volatility(closes)


def test_history_sorting_cutoff_and_duplicates():
    data = {
        "symbol": "_SPX",
        "data": [
            {"date": "2026-09-25", "close": "101"},
            {"date": "2026-09-24", "close": "100"},
        ],
    }
    assert parse_spx_history(data, end_date=date(2026, 9, 24)) == [
        (date(2026, 9, 24), 100)
    ]
    data["data"].append(data["data"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_spx_history(data)


def test_comparison_excludes_future_data_and_recovers_mid():
    snapshot = parse_spx_snapshot(payload())
    days = np.busday_offset(np.datetime64("2026-09-25"), np.arange(-252, 1))
    closes = 100 * np.exp(0.005 * np.sin(np.arange(253)))
    history = [
        (date.fromisoformat(str(day)), float(close)) for day, close in zip(days, closes)
    ]
    result = compare_market(snapshot, history, rate=0.03, dividend_yield=0.01)
    future = history + [(date(2026, 9, 28), 10000)]
    assert compare_market(snapshot, future, rate=0.03, dividend_yield=0.01) == result
    assert abs(result["iv_price_error"]) < 1e-10
    assert result["implied_volatility"] > 0
