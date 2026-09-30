"""Black-Scholes reference prices, parity and boundary conditions."""

from math import exp

import pytest

from derivative_pricer import black_scholes_price


def price(**overrides):
    inputs = dict(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)
    inputs.update(overrides)
    return black_scholes_price(**inputs)


@pytest.mark.parametrize(
    "option_type, expected",
    [
        ("call", 10.450583572185565),
        ("put", 5.573526022256971),
    ],
)
def test_reference_prices(option_type, expected):
    assert price(option_type=option_type) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "spot, rate, dividend_yield, maturity, volatility",
    [
        pytest.param(60, 0.05, 0, 1, 0.2, id="call-OTM"),
        pytest.param(100, 0.05, 0, 1, 0.2, id="ATM"),
        pytest.param(150, 0.05, 0, 1, 0.2, id="call-ITM"),
        pytest.param(100, 0.03, 0.02, 1, 0.2, id="dividends"),
        pytest.param(100, -0.02, 0.04, 1, 0.2, id="negative-rate"),
        pytest.param(100, 0.05, 0, 1e-8, 0.2, id="short-maturity"),
        pytest.param(100, 0.05, 0, 2, 1e-8, id="low-volatility"),
        pytest.param(100, 0.05, 0, 1, 0, id="zero-volatility"),
    ],
)
def test_parity_and_no_arbitrage_bounds(
    spot, rate, dividend_yield, maturity, volatility
):
    inputs = dict(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        maturity=maturity,
        volatility=volatility,
    )
    call = price(**inputs, option_type="call")
    put = price(**inputs, option_type="put")
    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = 100 * exp(-rate * maturity)
    forward_value = discounted_spot - discounted_strike

    assert call - put == pytest.approx(forward_value, abs=1e-11)
    # Allow for floating-point roundoff at the bounds.
    assert max(forward_value, 0) - 1e-12 <= call <= discounted_spot + 1e-12
    assert max(-forward_value, 0) - 1e-12 <= put <= discounted_strike + 1e-12


def test_call_monotonicity():
    assert price(spot=80) < price(spot=100) < price(spot=120)
    assert price(strike=80) > price(strike=100) > price(strike=120)
    assert price(volatility=0.1) < price(volatility=0.2) < price(volatility=0.4)
    assert price(dividend_yield=0) > price(dividend_yield=0.03)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("spot", [0, 80, 100, 120])
def test_expiry(option_type, spot):
    sign = 1 if option_type == "call" else -1
    assert price(spot=spot, maturity=0, option_type=option_type) == max(
        sign * (spot - 100), 0
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_volatility_is_discounted_forward_payoff(option_type):
    sign = 1 if option_type == "call" else -1
    expected = exp(-0.05) * max(sign * (100 * exp(0.05 - 0.02) - 100), 0)
    assert price(
        volatility=0, dividend_yield=0.02, option_type=option_type
    ) == pytest.approx(expected)


def test_zero_spot_and_strike():
    assert price(spot=0) == 0
    assert price(spot=0, option_type="put") == pytest.approx(100 * exp(-0.05))
    assert price(strike=0, dividend_yield=0.02) == pytest.approx(100 * exp(-0.02))
    assert price(strike=0, option_type="put") == 0
    assert price(spot=0, strike=0) == 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_short_maturity_atm_limit(option_type):
    assert price(maturity=1e-12, option_type=option_type) == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize(
    "name", ["spot", "strike", "maturity", "rate", "volatility", "dividend_yield"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        price(**{name: value})


@pytest.mark.parametrize("name", ["spot", "strike", "maturity", "volatility"])
def test_negative_inputs_rejected(name):
    with pytest.raises(ValueError, match=name):
        price(**{name: -0.01})


def test_invalid_option_type_rejected_even_at_expiry():
    with pytest.raises(ValueError, match="option_type"):
        price(option_type="CALL", maturity=0)
