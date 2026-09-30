# Derivatives Pricing Engine

European option pricing and risk analysis in Python: Black-Scholes valuation,
analytical Greeks, implied volatility, Monte Carlo simulation and discrete delta
hedging. Includes numerical convergence experiments and an optional SPX market-data adapter.

## Installation

Python 3.11 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

```bash
python examples/pricing_demo.py
python -m pytest -q
```

The example prices a call and put, computes sensitivities, recovers implied
volatility, estimates the price by Monte Carlo and simulates a daily delta hedge.
Change `inputs` and `volatility` in the script to evaluate another scenario.

```python
from derivative_pricer import black_scholes_price, implied_volatility

inputs = dict(spot=100, strike=100, maturity=1, rate=0.05)
price = black_scholes_price(**inputs, volatility=0.20)
result = implied_volatility(market_price=price, **inputs)

print(price)              # 10.450583572185565
print(result.volatility) # 0.20
```

## Methods

| Module | Purpose |
|---|---|
| `black_scholes.py` | European call/put valuation with continuous dividends |
| `greeks.py` | Analytical Delta, Gamma, Vega, Theta and Rho |
| `finite_differences.py` | Central-difference estimates of the Greeks |
| `implied_vol.py` | Bracketed Newton-Raphson with bisection safeguards |
| `monte_carlo.py` | Exact terminal GBM simulation, standard error and confidence interval |
| `hedging.py` | Self-financing stock/cash hedge of a short call |
| `market_data.py` | Delayed SPX quotes, historical volatility and model comparison |

Pricing logic lives in `src/derivative_pricer/`. Examples and the notebook import
the package; NumPy and SciPy provide numerical primitives.

## Conventions

- Prices are per unit of underlying. Maturity is in years; rates, yields and annualized volatility are decimals.
- Interest rates and dividend yields are constant and continuously compounded. Negative rates are supported.
- Vega and Rho are raw derivatives: multiply by `0.01` for a one-percentage-point change.
- Theta is calendar decay per year, `-dV/dT`. Dividing by 365 gives an approximate calendar-day effect.
- Finite-difference bumps are absolute, in the corresponding parameter units.
- Prices handle zero maturity, volatility, spot and strike explicitly. Greeks require these inputs to be strictly positive.

The model assumes lognormal dynamics, European exercise and frictionless trading.
Extreme inputs beyond floating-point range are outside the supported numerical domain.

## Numerical analysis

```bash
python examples/greeks_demo.py --convergence
jupyter notebook notebooks/analysis.ipynb
```

Run all notebook cells to generate:

- Greeks versus spot and finite-difference errors for bumps from 1e-1 to 1e-8.
- Implied-volatility recovery across strikes, maturities and starting guesses.
- Monte Carlo convergence from 100 to 1,000,000 paths.
- Hedging P&L for daily, weekly and monthly rebalancing, with realized volatility of 15%, 20% and 25%.

Central differences trade truncation error against roundoff. Small Vega can make
implied volatility poorly identified; the solver checks price bounds and reports
convergence failures. The exact lower price bound returns zero volatility by
convention. Monte Carlo confidence intervals use a normal approximation and can
be unreliable when few nonzero payoffs are observed. Standard error scales as
N^-1/2; a particular realized pricing error need not decrease monotonically.

The hedge uses a fixed implied volatility for pricing and Delta. Cash accrues
interest and dividends are reinvested in stock. Default price drift is r-q.
Daily/weekly/monthly correspond to rebalancing every 1/5/21 steps on a 252-step
year, followed by liquidation at expiry. P&L is measured at expiry, without
transaction costs, funding spreads or slippage.

## Results

For S = K = 100, T = 1, r = 5%, q = 0 and volatility = 20%:

| Quantity | Value |
|---|---:|
| Black-Scholes call | 10.450584 |
| Black-Scholes put | 5.573526 |
| Monte Carlo call, 1,000,000 paths, seed 42 | 10.453195 |
| Monte Carlo standard error | 0.014731 |

Daily hedging with implied volatility of 20%, r = 3%, 10,000 paths and seed 123
(other inputs unchanged) gives mean terminal P&L of approximately +1.991, +0.011
and -1.978 for realized volatility of 15%, 20% and 25%, respectively.
More frequent rebalancing reduces discretization risk but does not remove a
volatility mismatch. Results are reproducible through the notebook; minor
floating-point differences may occur across library versions.

## Tests

```bash
python -m pytest -q
python -m pytest tests/test_black_scholes.py -v
python -m pytest tests/test_black_scholes.py::test_call_monotonicity -v
```

Tests cover reference prices, put-call parity, bounds, limiting cases,
analytical versus numerical Greeks, IV recovery and failure modes, Monte Carlo
statistics, hedge accounting and market-data validation. Named scenarios cover
moneyness, short/long maturities, dividends and negative rates. Simulations use
fixed seeds. Market-data tests use offline fixtures and require no network access.

## Market data

```bash
python -m derivative_pricer.market_data --rate 0.03 --dividend-yield 0.012
```

The adapter retrieves delayed SPX quotes and daily history from
[Cboe](https://www.cboe.com/tradable-products/sp-500/spx-options), selects a
European SPXW call near ATM and 30 days to expiry, and compares its midpoint
with Black-Scholes using historical volatility. Historical volatility uses
252 daily log returns, annualized by sqrt(252).

The supplied rate and dividend yield are assumptions, not fetched market curves.
`data/market_snapshot.json` records sources, timestamps, inputs and results.
The notebook reads this file when available and does not fetch data automatically.

Quotes may be asynchronous. Valuation uses the last underlying trade time as a
proxy; maturity uses ACT/365 with an approximate 16:00 New York expiry, excluding
early closes. Midpoints are not guaranteed executable. Historical volatility is
not a forecast of implied volatility. Public provider feeds may change or become
unavailable.

## Repository layout

```text
src/derivative_pricer/   Pricing and risk modules
tests/                  Offline test suite
examples/               Pricing and Greeks examples
notebooks/analysis.ipynb Numerical experiments
```

Generated figures, downloaded snapshots, caches and virtual environments are
excluded from Git. Notebook outputs are cleared to keep diffs small.
American options, stochastic volatility, multi-asset models and autocalls are
outside the scope of this project.
