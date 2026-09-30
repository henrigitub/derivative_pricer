"""Self-financing hedge accounting and P&L behavior."""

from math import exp

import numpy as np
import pytest

from derivative_pricer import (
    analytical_greeks,
    black_scholes_price,
    simulate_delta_hedging,
)


def test_single_period_accounting_includes_dividends_and_interest():
    result = simulate_delta_hedging(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        implied_volatility=0.2,
        realized_volatility=0,
        dividend_yield=0.02,
        n_steps=1,
        n_paths=2,
        drift=0.03,
        seed=1,
    )
    inputs = dict(
        spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2, dividend_yield=0.02
    )
    premium = black_scholes_price(**inputs)
    shares = analytical_greeks(**inputs).delta
    # Dividends are reinvested in shares; cash accrues interest.
    stock = shares * exp(0.02) * 100 * exp(0.03)
    cash = (premium - shares * 100) * exp(0.05)
    expected = stock + cash - max(100 * exp(0.03) - 100, 0)
    np.testing.assert_allclose(result.pnl, expected, atol=1e-12)
    np.testing.assert_allclose(result.terminal_cash, cash)


def test_matched_vol_more_frequent_hedging_reduces_dispersion():
    inputs = dict(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.03,
        implied_volatility=0.2,
        realized_volatility=0.2,
        n_paths=8000,
        seed=12,
    )
    daily = simulate_delta_hedging(**inputs)
    monthly = simulate_delta_hedging(**inputs, rebalance_every=21)
    np.testing.assert_array_equal(daily.terminal_spot, monthly.terminal_spot)
    assert daily.pnl.std() < monthly.pnl.std() / 2
    assert abs(daily.pnl.mean()) < 4 * daily.pnl.std(ddof=1) / np.sqrt(len(daily.pnl))
    np.testing.assert_allclose(
        daily.pnl,
        daily.terminal_stock
        + daily.terminal_cash
        - np.maximum(daily.terminal_spot - 100, 0),
    )


def test_short_call_earns_when_realized_vol_is_lower():
    inputs = dict(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.03,
        implied_volatility=0.2,
        n_paths=4000,
        seed=7,
    )
    low = simulate_delta_hedging(**inputs, realized_volatility=0.15)
    high = simulate_delta_hedging(**inputs, realized_volatility=0.25)
    assert low.pnl.mean() > 1
    assert high.pnl.mean() < -1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"realized_volatility": -0.1},
        {"n_steps": 0},
        {"rebalance_every": 1.5},
        {"drift": float("nan")},
    ],
)
def test_invalid_hedge_settings(kwargs):
    inputs = dict(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.03,
        implied_volatility=0.2,
        realized_volatility=0.2,
    )
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        simulate_delta_hedging(**inputs)
