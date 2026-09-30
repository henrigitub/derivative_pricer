"""Run pricing, risk and hedging examples with a shared market scenario."""

from math import exp

from derivative_pricer import (
    analytical_greeks,
    black_scholes_price,
    implied_volatility,
    monte_carlo_price,
    simulate_delta_hedging,
)


def main() -> None:
    inputs = dict(spot=100, strike=100, maturity=1, rate=0.05)
    volatility = 0.20

    call = black_scholes_price(**inputs, volatility=volatility, option_type="call")
    put = black_scholes_price(**inputs, volatility=volatility, option_type="put")
    parity = inputs["spot"] - inputs["strike"] * exp(
        -inputs["rate"] * inputs["maturity"]
    )
    print(f"Call: {call:.6f} | Put: {put:.6f}")
    print(f"Put-call parity residual: {call - put - parity:.2e}")

    greeks = analytical_greeks(**inputs, volatility=volatility)
    print(f"Delta: {greeks.delta:.6f} | Gamma: {greeks.gamma:.6f}")
    print(f"Vega per volatility point: {greeks.vega * 0.01:.6f}")
    print(f"Theta per calendar day: {greeks.theta / 365:.6f}")
    print(f"Rho per rate point: {greeks.rho * 0.01:.6f}")

    iv = implied_volatility(market_price=call, **inputs, initial_volatility=0.10)
    print(f"Recovered volatility: {iv.volatility:.2%} in {iv.iterations} iterations")

    mc = monte_carlo_price(**inputs, volatility=volatility, n_paths=100_000, seed=42)
    low, high = mc.confidence_interval
    print(f"Monte Carlo: {mc.price:.6f} | Standard error: {mc.standard_error:.6f}")
    print(f"95% confidence interval: [{low:.6f}, {high:.6f}]")

    hedge = simulate_delta_hedging(
        **inputs,
        implied_volatility=volatility,
        realized_volatility=volatility,
        n_paths=5_000,
        n_steps=252,
        rebalance_every=1,
        seed=123,
    )
    print(f"Mean hedging P&L: {hedge.pnl.mean():.6f}")
    print(f"Hedging P&L standard deviation: {hedge.pnl.std(ddof=1):.6f}")


if __name__ == "__main__":
    main()
