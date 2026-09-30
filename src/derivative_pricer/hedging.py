"""Self-financing discrete delta hedge of one short European call."""

from dataclasses import dataclass
from math import exp, isfinite, sqrt

import numpy as np
from scipy.special import ndtr

from .black_scholes import black_scholes_price
from .greeks import _validate_greek_inputs


@dataclass(frozen=True)
class HedgingResult:
    pnl: np.ndarray
    terminal_spot: np.ndarray
    initial_premium: float
    terminal_stock: np.ndarray
    terminal_cash: np.ndarray


def simulate_delta_hedging(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    implied_volatility: float,
    realized_volatility: float,
    dividend_yield: float = 0.0,
    *,
    n_paths: int = 10_000,
    n_steps: int = 252,
    rebalance_every: int = 1,
    seed: int | None = None,
    drift: float | None = None,
) -> HedgingResult:
    """Return terminal P&L = stock + cash - call payoff, per option unit.

    Premium and hedge Delta use implied volatility; paths use realized
    volatility. Default price drift is r-q (risk-neutral); a supplied drift
    is the annual price drift, excluding dividends. Interest accrues at r.
    Continuous dividends are reinvested into stock between rebalances.
    Rebalance every k base steps, excluding expiry; liquidate at expiry.
    Same seed/base grid gives identical paths across hedge frequencies.
    No costs, funding spread or slippage; P&L is not discounted.
    """
    _validate_greek_inputs(
        spot, strike, maturity, rate, implied_volatility, "call", dividend_yield
    )
    if not isfinite(realized_volatility) or realized_volatility < 0:
        raise ValueError("realized_volatility must be finite and nonnegative")
    for name, value in (
        ("n_paths", n_paths),
        ("n_steps", n_steps),
        ("rebalance_every", rebalance_every),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if drift is None:
        drift = rate - dividend_yield
    if not isfinite(drift):
        raise ValueError("drift must be finite")

    def call_delta(current_spot: np.ndarray, remaining: float) -> np.ndarray:
        # Evaluate the same Black-Scholes Delta on every simulated path.
        total_volatility = implied_volatility * sqrt(remaining)
        log_moneyness = np.log(current_spot / strike)
        carry = (rate - dividend_yield) * remaining
        d1 = (log_moneyness + carry) / total_volatility + total_volatility / 2
        return exp(-dividend_yield * remaining) * ndtr(d1)

    premium = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=implied_volatility,
        option_type="call",
        dividend_yield=dividend_yield,
    )
    prices = np.full(n_paths, spot, dtype=float)
    shares = call_delta(prices, maturity)
    cash = premium - shares * prices
    dt = maturity / n_steps
    drift_per_step = (drift - realized_volatility**2 / 2) * dt
    volatility_per_step = realized_volatility * sqrt(dt)
    cash_growth = exp(rate * dt)
    dividend_growth = exp(dividend_yield * dt)
    rng = np.random.default_rng(seed)

    for step in range(1, n_steps + 1):
        normal_draws = rng.standard_normal(n_paths)
        prices *= np.exp(drift_per_step + volatility_per_step * normal_draws)
        cash *= cash_growth
        shares *= dividend_growth

        is_rebalance_date = step % rebalance_every == 0
        before_expiry = step < n_steps
        if before_expiry and is_rebalance_date:
            remaining = maturity * (n_steps - step) / n_steps
            new_shares = call_delta(prices, remaining)
            shares_to_buy = new_shares - shares
            trade_cost = shares_to_buy * prices
            cash -= trade_cost  # A sale has negative cost and credits cash.
            shares = new_shares

    terminal_stock = shares * prices
    call_payoff = np.maximum(prices - strike, 0.0)
    pnl = terminal_stock + cash - call_payoff
    return HedgingResult(
        pnl=pnl,
        terminal_spot=prices,
        initial_premium=premium,
        terminal_stock=terminal_stock,
        terminal_cash=cash,
    )
