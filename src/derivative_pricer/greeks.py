"""Analytical sensitivities of European options under Black-Scholes."""

from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Literal

from scipy.special import ndtr

from .black_scholes import _validate_inputs


@dataclass(frozen=True)
class Greeks:
    """Raw derivatives: Vega/Rho per 1.0 change, Theta per year elapsed.

    Multiply Vega or Rho by 0.01 for a one-percentage-point change.
    Divide Theta by 365 for an approximate calendar-day decay.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate_greek_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float,
) -> None:
    _validate_inputs(
        spot, strike, maturity, rate, volatility, option_type, dividend_yield
    )
    # Boundary prices exist, but some derivatives there are singular or one-sided.
    for name, value in (
        ("spot", spot),
        ("strike", strike),
        ("maturity", maturity),
        ("volatility", volatility),
    ):
        if value == 0:
            raise ValueError(f"{name} must be strictly positive for Greeks")


def analytical_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
) -> Greeks:
    """Return Delta, Gamma, Vega, calendar Theta and Rho for scalar inputs.

    Uses the same units as black_scholes_price, but requires strictly positive
    spot, strike, maturity and volatility. Theta = -dV/dT, with all market
    parameters held constant. Boundary Greeks are excluded.
    """
    _validate_greek_inputs(
        spot, strike, maturity, rate, volatility, option_type, dividend_yield
    )
    sqrt_t = sqrt(maturity)
    total_volatility = volatility * sqrt_t
    log_moneyness = log(spot) - log(strike)
    carry = (rate - dividend_yield) * maturity
    d1 = (log_moneyness + carry) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    discount_q = exp(-dividend_yield * maturity)
    discount_r = exp(-rate * maturity)
    density = exp(-0.5 * d1 * d1) / sqrt(2 * pi)  # phi(d1), not the CDF N(d1).

    # Gamma and Vega are identical for calls and puts with the same inputs.
    gamma = discount_q * density / (spot * volatility * sqrt_t)
    vega = spot * discount_q * density * sqrt_t
    time_decay = -spot * discount_q * density * volatility / (2 * sqrt_t)

    if option_type == "call":
        delta = discount_q * ndtr(d1)
        theta = (
            time_decay
            - rate * strike * discount_r * ndtr(d2)
            + dividend_yield * spot * discount_q * ndtr(d1)
        )
        rho = strike * maturity * discount_r * ndtr(d2)
    else:
        delta = -discount_q * ndtr(-d1)
        theta = (
            time_decay
            + rate * strike * discount_r * ndtr(-d2)
            - dividend_yield * spot * discount_q * ndtr(-d1)
        )
        rho = -strike * maturity * discount_r * ndtr(-d2)

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
