"""Black-Scholes valuation with a continuous dividend yield."""

from math import exp, isfinite, log, sqrt
from typing import Literal

from scipy.special import ndtr


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Return the price per unit of underlying of a European call or put.

    Inputs are finite real scalars. Maturity is in years; rate, dividend_yield
    and annualized volatility are decimals (0.20 means 20%). Rates and dividend
    yields are continuously compounded and may be negative. Spot, strike,
    maturity and volatility must not be negative.

    At expiry, returns intrinsic value. At zero
    volatility, returns the discounted deterministic payoff.

    Raises ValueError for nonfinite inputs, invalid signs or option_type.
    """
    _validate_inputs(
        spot, strike, maturity, rate, volatility, option_type, dividend_yield
    )

    if maturity == 0:
        if option_type == "call":
            return float(max(spot - strike, 0.0))
        return float(max(strike - spot, 0.0))

    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)
    if volatility == 0 or spot == 0 or strike == 0:
        if option_type == "call":
            return float(max(discounted_spot - discounted_strike, 0.0))
        return float(max(discounted_strike - discounted_spot, 0.0))

    total_volatility = volatility * sqrt(maturity)
    log_moneyness = log(spot) - log(strike)
    carry = (rate - dividend_yield) * maturity
    d1 = (log_moneyness + carry) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility

    if option_type == "call":
        price = discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2)
    else:
        price = discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1)
    return float(price)


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float,
) -> None:
    """Shared scalar input validation for pricing and risk functions."""
    values = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
    }
    for name, value in values.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in ("spot", "strike", "maturity", "volatility"):
        if values[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
