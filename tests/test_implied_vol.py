"""Implied-volatility recovery and solver failure modes."""

import pytest

from derivative_pricer import black_scholes_price, implied_volatility


@pytest.mark.parametrize(
    "option_type, strike, maturity, initial",
    [
        pytest.param("call", 80, 1, 0.2, id="call-ITM"),
        pytest.param("call", 100, 1, 0.2, id="call-ATM"),
        pytest.param("call", 120, 1, 0.2, id="call-OTM"),
        pytest.param("put", 80, 1, 0.2, id="put-OTM"),
        pytest.param("put", 100, 1, 0.2, id="put-ATM"),
        pytest.param("put", 120, 1, 0.2, id="put-ITM"),
        pytest.param("call", 100, 0.01, 0.2, id="short-maturity"),
        pytest.param("put", 120, 5, 0.2, id="long-maturity"),
        pytest.param("call", 100, 1, 0.001, id="low-initial-guess"),
        pytest.param("put", 100, 1, 3.0, id="high-initial-guess"),
    ],
)
def test_recover_known_volatility(option_type, strike, maturity, initial):
    inputs = dict(
        spot=100,
        strike=strike,
        maturity=maturity,
        rate=-0.01,
        option_type=option_type,
        dividend_yield=0.02,
    )

    price = black_scholes_price(**inputs, volatility=0.27)

    result = implied_volatility(
        market_price=price, **inputs, initial_volatility=initial
    )
    assert result.volatility == pytest.approx(0.27, abs=1e-8)
    assert abs(result.price_error) <= 1e-10


def test_low_vega_uses_bisection():
    inputs = dict(spot=100, strike=120, maturity=0.1, rate=0.02)
    price = black_scholes_price(**inputs, volatility=0.4)
    result = implied_volatility(market_price=price, **inputs, initial_volatility=0.001)
    assert result.bisection_steps > 0
    assert result.volatility == pytest.approx(0.4, abs=1e-8)


@pytest.mark.parametrize("price", [-1, 101, 100, float("nan"), float("inf")])
def test_invalid_prices(price):
    with pytest.raises(ValueError):
        implied_volatility(market_price=price, spot=100, strike=100, maturity=1, rate=0)


def test_lower_bound_and_nonidentifiable_expiry():
    assert (
        implied_volatility(
            market_price=0, spot=100, strike=100, maturity=1, rate=0
        ).volatility
        == 0
    )
    with pytest.raises(ValueError):
        implied_volatility(market_price=0, spot=100, strike=100, maturity=0, rate=0)


def test_nonconvergence_is_explicit():
    with pytest.raises(RuntimeError, match="did not converge"):
        implied_volatility(
            market_price=10,
            spot=100,
            strike=100,
            maturity=1,
            rate=0.05,
            initial_volatility=2,
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price_tolerance": 0},
        {"volatility_tolerance": -1},
        {"max_iterations": 1.5},
        {"initial_volatility": 0},
    ],
)
def test_invalid_solver_settings(kwargs):
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=10, spot=100, strike=100, maturity=1, rate=0.05, **kwargs
        )
