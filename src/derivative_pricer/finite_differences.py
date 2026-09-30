"""Estimate Greeks by repricing with symmetric parameter perturbations."""

from math import isfinite
from typing import Literal

from .black_scholes import black_scholes_price
from .greeks import Greeks, _validate_greek_inputs


def finite_difference_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
    *,
    spot_bump: float = 0.01,
    volatility_bump: float = 1e-4,
    maturity_bump: float = 1e-5,
    rate_bump: float = 1e-5,
) -> Greeks:
    """Return central-difference estimates with the analytical Greeks' units.

    Bumps are absolute: spot in currency, maturity in years, rate/volatility
    in decimals. All bumps must be finite and positive. Spot, volatility and
    maturity bumps must be smaller than their base values so both evaluations
    stay in the interior of the domain. Defaults suit the example, not every
    instrument; reduce them explicitly for very short maturities, etc.
    """
    _validate_greek_inputs(
        spot, strike, maturity, rate, volatility, option_type, dividend_yield
    )
    bumps = {
        "spot": spot_bump,
        "volatility": volatility_bump,
        "maturity": maturity_bump,
        "rate": rate_bump,
    }
    inputs = dict(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    for name, bump in bumps.items():
        if not isfinite(bump) or bump <= 0:
            raise ValueError(f"{name}_bump must be finite and positive")
        if name != "rate" and bump >= inputs[name]:
            raise ValueError(f"{name}_bump must be smaller than {name}")
        if inputs[name] + bump == inputs[name] or inputs[name] - bump == inputs[name]:
            raise ValueError(f"{name}_bump is too small for floating-point arithmetic")

    def repriced(parameter: str, shift: float) -> float:
        bumped_inputs = inputs.copy()  # Keep the base scenario unchanged.
        bumped_inputs[parameter] += shift
        return black_scholes_price(**bumped_inputs)

    base_price = black_scholes_price(**inputs)
    price_spot_up = repriced("spot", spot_bump)
    price_spot_down = repriced("spot", -spot_bump)
    delta = (price_spot_up - price_spot_down) / (2 * spot_bump)
    gamma = (price_spot_up - 2 * base_price + price_spot_down) / spot_bump**2

    price_volatility_up = repriced("volatility", volatility_bump)
    price_volatility_down = repriced("volatility", -volatility_bump)
    vega = (price_volatility_up - price_volatility_down) / (2 * volatility_bump)

    price_maturity_up = repriced("maturity", maturity_bump)
    price_maturity_down = repriced("maturity", -maturity_bump)
    # Advancing calendar time reduces remaining maturity: Theta = -dV/dT.
    theta = -(price_maturity_up - price_maturity_down) / (2 * maturity_bump)

    price_rate_up = repriced("rate", rate_bump)
    price_rate_down = repriced("rate", -rate_bump)
    rho = (price_rate_up - price_rate_down) / (2 * rate_bump)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
