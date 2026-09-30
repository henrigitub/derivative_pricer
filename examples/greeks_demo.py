"""Compare Greeks; add --convergence to plot error versus absolute bump."""

import argparse
from pathlib import Path

from derivative_pricer import analytical_greeks, finite_difference_greeks


def plot_bump_errors(inputs: dict, output: Path) -> None:
    """Study one scenario using eight absolute bumps, with raw Greek units."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = ("delta", "gamma", "vega", "theta", "rho")
    bumps = np.logspace(-1, -8, 8)
    exact = analytical_greeks(**inputs)
    errors = {name: [] for name in names}
    print("\nAbsolute errors (raw Greek units):")
    print(f"{'Bump':>10}" + "".join(f"{name:>14}" for name in names))
    for bump in bumps:
        estimated = finite_difference_greeks(
            **inputs,
            spot_bump=bump,
            volatility_bump=bump,
            maturity_bump=bump,
            rate_bump=bump,
        )
        for name in names:
            errors[name].append(abs(getattr(estimated, name) - getattr(exact, name)))
        print(f"{bump:10.0e}" + "".join(f"{errors[name][-1]:14.3e}" for name in names))

    labels = {
        "delta": "Spot bump (currency units)",
        "gamma": "Spot bump (currency units)",
        "vega": "Volatility bump (decimal)",
        "theta": "Remaining maturity bump (years)",
        "rho": "Rate bump (decimal)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    fig.suptitle("Central differences: error versus absolute bump", fontsize=15)
    for ax, name in zip(axes.flat, names):
        values = np.asarray(errors[name])
        # An exact floating-point match has zero error, undefined on a log axis.
        visible = np.where(values > 0, values, np.nan)
        ax.loglog(bumps, visible, "o-", color="#2563eb", linewidth=1.6)
        ax.set(
            title=name.capitalize(),
            xlabel=labels[name],
            ylabel="Absolute error (raw units)",
        )
        ax.grid(True, which="major", alpha=0.25)
        if np.any(values == 0):
            ax.text(
                0.03, 0.03, "Zero errors omitted", transform=ax.transAxes, fontsize=8
            )
        best = int(np.argmin(values))
        print(
            f"{name:>5}: smallest sampled error at h={bumps[best]:.0e}: {values[best]:.3e}"
        )

    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.95,
        "European call\nS = K = 100, T = 1 year\nr = 5%, q = 0%, volatility = 20%\n\n"
        "Large bumps: truncation error\nSmall bumps: roundoff / cancellation\n\n"
        "Bumps have different units across Greeks.\nEach panel has its own vertical scale.\n"
        "Sampled minima are scenario-dependent.",
        va="top",
        fontsize=10,
        linespacing=1.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"\nSaved figure: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convergence", action="store_true", help="Study bumps from 1e-1 to 1e-8"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/finite_difference_errors.png")
    )
    args = parser.parse_args()
    inputs = dict(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.20)
    analytical = analytical_greeks(**inputs)
    numerical = finite_difference_greeks(**inputs)
    print(f"{'Greek':<8} {'Analytical':>14} {'Finite diff.':>14} {'Abs. error':>14}")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        exact = getattr(analytical, name)
        estimate = getattr(numerical, name)
        print(f"{name:<8} {exact:14.8f} {estimate:14.8f} {abs(exact - estimate):14.3e}")
    print(f"\nVega per volatility point: {analytical.vega * 0.01:.6f}")
    print(f"Theta per calendar day:   {analytical.theta / 365:.6f}")
    print(f"Rho per rate point:       {analytical.rho * 0.01:.6f}")
    if args.convergence:
        plot_bump_errors(inputs, args.output)


if __name__ == "__main__":
    main()
