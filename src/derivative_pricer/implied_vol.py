"""Implied volatility through bracketed Newton steps and bisection safeguards."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Literal

from .black_scholes import black_scholes_price, _validate_inputs
from .greeks import analytical_greeks


@dataclass(frozen=True)
class ImpliedVolResult:
    volatility: float
    iterations: int
    price_error: float
    bisection_steps: int


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
    *,
    initial_volatility: float = 0.2,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> ImpliedVolResult:
    """Invert Black-Scholes, returning diagnostics or raising on failure.

    Tolerances are absolute (price units and decimal volatility). The price
    must lie between the zero-volatility value and the strict upper bound.
    The exact lower bound returns zero volatility by convention; at floating-
    point precision a deep ITM/OTM option may not identify volatility uniquely.
    Zero maturity/spot/strike are rejected because inversion is undefined.
    """
    _validate_inputs(
        spot, strike, maturity, rate, initial_volatility, option_type, dividend_yield
    )
    if min(spot, strike, maturity, initial_volatility) <= 0:
        raise ValueError(
            "spot, strike, maturity and initial_volatility must be positive"
        )
    if not isfinite(market_price) or market_price < 0:
        raise ValueError("market_price must be finite and nonnegative")
    for name, value in (
        ("price_tolerance", price_tolerance),
        ("volatility_tolerance", volatility_tolerance),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    inputs = dict(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    lower_price = black_scholes_price(**inputs, volatility=0)
    if option_type == "call":
        upper_price = spot * exp(-dividend_yield * maturity)
    else:
        upper_price = strike * exp(-rate * maturity)
    if not lower_price <= market_price < upper_price:
        raise ValueError(
            "market_price violates the finite-volatility no-arbitrage bounds"
        )
    if market_price == lower_price:
        return ImpliedVolResult(
            volatility=0.0, iterations=0, price_error=0.0, bisection_steps=0
        )

    low, high = 0.0, 1.0
    # A finite bracket is found before Newton begins; no unbounded loop.
    for _ in range(64):
        if black_scholes_price(**inputs, volatility=high) >= market_price:
            break
        high *= 2
    else:
        raise RuntimeError("Could not bracket implied volatility")
    volatility = min(initial_volatility, high)
    bisections = 0
    for iteration in range(1, max_iterations + 1):
        model_price = black_scholes_price(**inputs, volatility=volatility)
        error = model_price - market_price
        vega = analytical_greeks(**inputs, volatility=volatility).vega

        # A small price error alone may hide a large IV error when Vega is small.
        price_converged = abs(error) <= price_tolerance
        newton_step = error / vega if vega > 0 else float("inf")
        step_converged = abs(newton_step) <= volatility_tolerance
        bracket_converged = high - low <= volatility_tolerance
        if price_converged and (error == 0 or step_converged or bracket_converged):
            return ImpliedVolResult(
                volatility=float(volatility),
                iterations=iteration,
                price_error=float(error),
                bisection_steps=bisections,
            )
        if error > 0:
            high = volatility
        else:
            low = volatility
        if vega > 1e-14:
            candidate = volatility - newton_step
        else:
            candidate = float("nan")
        newton_step_is_valid = isfinite(candidate) and low < candidate < high
        if not newton_step_is_valid:
            candidate = (low + high) / 2
            bisections += 1
        volatility = candidate
    raise RuntimeError(
        f"Implied volatility did not converge after {max_iterations} iterations"
    )
