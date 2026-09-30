"""Monte Carlo pricing, sampling error and reproducibility."""

import numpy as np
import pytest

from derivative_pricer import black_scholes_price, monte_carlo_price


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("rate,q", [(0.05, 0), (-0.01, 0.03)])
def test_price_consistent_with_black_scholes(option_type, rate, q):
    inputs = dict(
        spot=100,
        strike=105,
        maturity=1.3,
        rate=rate,
        volatility=0.25,
        dividend_yield=q,
        option_type=option_type,
    )
    result = monte_carlo_price(**inputs, n_paths=200_000, seed=17)
    exact = black_scholes_price(**inputs)
    # A statistical check with a fixed seed; do not demand every 95% CI hit.
    assert abs(result.price - exact) < 4 * result.standard_error
    assert result.confidence_interval[0] < result.price < result.confidence_interval[1]


def test_manual_sample_statistics_and_reproducibility():
    result = monte_carlo_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.2,
        n_paths=1000,
        seed=3,
    )
    z = np.random.default_rng(3).standard_normal(1000)
    payoffs = np.exp(-0.05) * np.maximum(100 * np.exp(0.03 + 0.2 * z) - 100, 0)
    assert result.price == pytest.approx(payoffs.mean())
    assert result.standard_error == pytest.approx(payoffs.std(ddof=1) / np.sqrt(1000))
    assert result == monte_carlo_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.2,
        n_paths=1000,
        seed=3,
    )


@pytest.mark.parametrize("maturity,volatility", [(0, 0.2), (1, 0)])
def test_deterministic_limits(maturity, volatility):
    result = monte_carlo_price(
        spot=100, strike=90, maturity=maturity, rate=0.05, volatility=volatility
    )
    assert result.price == black_scholes_price(
        spot=100, strike=90, maturity=maturity, rate=0.05, volatility=volatility
    )
    assert result.standard_error == 0


def test_standard_error_scales_as_inverse_square_root():
    small = monte_carlo_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.2,
        n_paths=10_000,
        seed=5,
    )
    large = monte_carlo_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.2,
        n_paths=1_000_000,
        seed=5,
    )
    assert 9.5 < small.standard_error / large.standard_error < 10.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_paths": 1},
        {"n_paths": 3.5},
        {"confidence_level": 1},
        {"confidence_level": float("nan")},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        monte_carlo_price(
            spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2, **kwargs
        )
