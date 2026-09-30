"""Exact risk-neutral GBM terminal simulation for European payoffs."""

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Literal

import numpy as np
from scipy.special import ndtri

from .black_scholes import _validate_inputs, black_scholes_price


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_paths: int
    confidence_level: float


def monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
    *,
    n_paths: int = 100_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price with iid draws and a normal-approximation confidence interval.

    The sample standard deviation uses ddof=1. The confidence interval is
    asymptotic, not an exact guarantee, especially for rare nonzero payoffs.
    Zero empirical error can mean no payoff was observed, not zero true risk.
    """
    _validate_inputs(
        spot, strike, maturity, rate, volatility, option_type, dividend_yield
    )
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    if not isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if maturity == 0 or volatility == 0 or spot == 0:
        price = black_scholes_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        return MonteCarloResult(
            price=price,
            standard_error=0.0,
            confidence_interval=(price, price),
            n_paths=n_paths,
            confidence_level=confidence_level,
        )
    rng = np.random.default_rng(seed)
    normal_draws = rng.standard_normal(n_paths)
    drift_term = (rate - dividend_yield - volatility**2 / 2) * maturity
    diffusion_term = volatility * sqrt(maturity) * normal_draws
    terminal_spot = spot * np.exp(drift_term + diffusion_term)

    if option_type == "call":
        payoffs = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_spot, 0.0)
    discount_factor = exp(-rate * maturity)
    discounted_payoffs = discount_factor * payoffs

    price = float(discounted_payoffs.mean())
    payoff_standard_deviation = float(discounted_payoffs.std(ddof=1))
    standard_error = payoff_standard_deviation / sqrt(n_paths)
    tail_probability = (1 - confidence_level) / 2
    normal_quantile = float(-ndtri(tail_probability))
    margin = normal_quantile * standard_error
    confidence_interval = (price - margin, price + margin)
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        n_paths=n_paths,
        confidence_level=confidence_level,
    )
