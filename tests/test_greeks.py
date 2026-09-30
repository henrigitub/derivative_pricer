"""Analytical sensitivities and finite-difference checks."""

from math import exp

import pytest

from derivative_pricer import analytical_greeks, finite_difference_greeks

BASE = dict(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)


def test_call_reference_greeks():
    result = analytical_greeks(**BASE)
    assert result.delta == pytest.approx(0.6368306512)
    assert result.gamma == pytest.approx(0.01876201735)
    assert result.vega == pytest.approx(37.52403469)
    assert result.theta == pytest.approx(-6.414027546)
    assert result.rho == pytest.approx(53.23248155)


@pytest.mark.parametrize(
    "option_type, spot, maturity, volatility, rate, dividend_yield",
    [
        pytest.param("call", 80, 1, 0.2, 0.05, 0, id="call-OTM"),
        pytest.param("call", 100, 1, 0.2, 0.05, 0, id="call-ATM"),
        pytest.param("call", 120, 1, 0.2, 0.05, 0, id="call-ITM"),
        pytest.param("put", 80, 1, 0.2, 0.05, 0, id="put-ITM"),
        pytest.param("put", 100, 1, 0.2, 0.05, 0, id="put-ATM"),
        pytest.param("put", 120, 1, 0.2, 0.05, 0, id="put-OTM"),
        pytest.param("call", 100, 0.02, 0.35, 0.05, 0, id="short-maturity"),
        pytest.param("put", 120, 5, 0.1, 0.05, 0, id="long-maturity"),
        pytest.param(
            "call", 100, 1, 0.2, -0.01, 0.03, id="call-negative-rate-dividends"
        ),
        pytest.param("put", 100, 1, 0.2, -0.01, 0.03, id="put-negative-rate-dividends"),
    ],
)
def test_analytical_matches_repricing(
    option_type, spot, maturity, volatility, rate, dividend_yield
):
    inputs = dict(
        strike=100,
        option_type=option_type,
        spot=spot,
        maturity=maturity,
        volatility=volatility,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    exact = analytical_greeks(**inputs)
    numerical = finite_difference_greeks(
        **inputs,
        spot_bump=0.001,
        volatility_bump=1e-5,
        maturity_bump=1e-6,
        rate_bump=1e-6,
    )
    # Allow for finite-difference truncation and roundoff errors.
    assert numerical.delta == pytest.approx(exact.delta, rel=2e-5, abs=2e-7)
    assert numerical.gamma == pytest.approx(exact.gamma, rel=2e-5, abs=2e-7)
    assert numerical.vega == pytest.approx(exact.vega, rel=2e-5, abs=2e-7)
    assert numerical.theta == pytest.approx(exact.theta, rel=2e-5, abs=2e-7)
    assert numerical.rho == pytest.approx(exact.rho, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("rate, q", [(0.05, 0.02), (-0.01, 0.03)])
def test_greek_parity_identities(rate, q):
    inputs = BASE.copy()
    inputs.update(rate=rate, dividend_yield=q)
    call = analytical_greeks(**inputs)
    put = analytical_greeks(**inputs, option_type="put")
    assert call.delta - put.delta == pytest.approx(exp(-q))
    assert call.gamma == put.gamma
    assert call.vega == put.vega
    assert call.rho - put.rho == pytest.approx(100 * exp(-rate))
    assert call.theta - put.theta == pytest.approx(
        q * 100 * exp(-q) - rate * 100 * exp(-rate)
    )
    assert 0 < call.delta < exp(-q)
    assert -exp(-q) < put.delta < 0
    assert call.gamma > 0 and call.vega > 0
    assert call.rho > 0 and put.rho < 0


@pytest.mark.parametrize("function", [analytical_greeks, finite_difference_greeks])
@pytest.mark.parametrize("name", ["spot", "strike", "maturity", "volatility"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_greek_domain(function, name, value):
    inputs = BASE.copy()
    inputs[name] = value
    with pytest.raises(ValueError, match=name):
        function(**inputs)


@pytest.mark.parametrize(
    "name", ["spot_bump", "volatility_bump", "maturity_bump", "rate_bump"]
)
@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), 1e-30])
def test_invalid_bumps(name, value):
    with pytest.raises(ValueError, match=name):
        finite_difference_greeks(**BASE, **{name: value})


@pytest.mark.parametrize("name", ["spot", "volatility", "maturity"])
def test_bumps_cannot_touch_boundary(name):
    with pytest.raises(ValueError, match=f"{name}_bump"):
        finite_difference_greeks(**BASE, **{f"{name}_bump": BASE[name]})


def test_smaller_bump_improves_delta_before_roundoff_dominates():
    exact = analytical_greeks(**BASE).delta
    coarse = finite_difference_greeks(**BASE, spot_bump=2).delta
    fine = finite_difference_greeks(**BASE, spot_bump=1).delta
    assert 3.9 < abs(coarse - exact) / abs(fine - exact) < 4.1
