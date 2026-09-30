"""Public delayed Cboe SPX data; no network access during module import."""

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import json
from math import isfinite, sqrt
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np

from .black_scholes import black_scholes_price
from .implied_vol import implied_volatility

OPTIONS_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
HISTORY_URL = (
    "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_SPX.json"
)
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class OptionQuote:
    symbol: str
    expiry: date
    strike: float
    option_type: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class MarketSnapshot:
    spot: float
    as_of: datetime
    provider_timestamp: str
    retrieved_at: str
    options: tuple[OptionQuote, ...]


def _download_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "derivatives-pricing-engine/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_spx_snapshot(payload: dict) -> MarketSnapshot:
    """Parse PM-settled SPXW only; discard malformed or nonpositive markets.

    Cboe's naive underlying last_trade_time is interpreted in New York time.
    This is the valuation time proxy; bid/ask and spot need not be synchronous.
    Provider generation time is preserved verbatim rather than guessed as UTC.
    """
    data = payload["data"]
    if data.get("symbol") != "^SPX":
        raise ValueError("Expected SPX underlying")
    spot = float(data["current_price"])
    if not isfinite(spot) or spot <= 0:
        raise ValueError("Invalid SPX spot")
    as_of = datetime.fromisoformat(data["last_trade_time"])
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=NEW_YORK)
    options = []
    for row in data["options"]:
        match = re.fullmatch(r"SPXW(\d{6})([CP])(\d{8})", row.get("option", ""))
        if not match:
            continue
        try:
            expiry = datetime.strptime(match[1], "%y%m%d").date()
            bid, ask = float(row["bid"]), float(row["ask"])
            strike = int(match[3]) / 1000
        except (KeyError, TypeError, ValueError):
            continue
        if not (isfinite(bid) and isfinite(ask) and 0 < bid <= ask and strike > 0):
            continue
        options.append(
            OptionQuote(
                row["option"],
                expiry,
                strike,
                "call" if match[2] == "C" else "put",
                bid,
                ask,
            )
        )
    if not options:
        raise ValueError("No usable SPXW bid/ask quotes")
    return MarketSnapshot(
        spot,
        as_of,
        str(payload["timestamp"]),
        datetime.now(timezone.utc).isoformat(),
        tuple(options),
    )


def fetch_spx_snapshot() -> MarketSnapshot:
    """Fetch delayed European index options; provider failures propagate."""
    return parse_spx_snapshot(_download_json(OPTIONS_URL))


def parse_spx_history(
    payload: dict, *, end_date: date | None = None
) -> list[tuple[date, float]]:
    if payload.get("symbol") not in ("_SPX", "^SPX"):
        raise ValueError("Expected SPX history")
    rows = []
    for item in payload["data"]:
        day, close = date.fromisoformat(item["date"]), float(item["close"])
        if end_date is not None and day > end_date:
            continue
        if not isfinite(close) or close <= 0:
            raise ValueError("Historical closes must be finite and positive")
        rows.append((day, close))
    rows.sort()
    if len({day for day, _ in rows}) != len(rows):
        raise ValueError("Duplicate history dates")
    return rows


def fetch_spx_history(*, end_date: date | None = None) -> list[tuple[date, float]]:
    return parse_spx_history(_download_json(HISTORY_URL), end_date=end_date)


def historical_volatility(closes, *, periods_per_year: int = 252) -> float:
    """Annualized sample std of consecutive daily log returns (ddof=1).

    Caller supplies sorted consecutive trading-day closes, with no gaps.
    Price-index returns are used here, not total returns or a forecast of IV.
    """
    values = np.asarray(closes, dtype=float)
    if (
        values.ndim != 1
        or values.size < 3
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("At least three positive finite closes are required")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    log_returns = np.diff(np.log(values))
    daily_volatility = np.std(log_returns, ddof=1)
    annualized_volatility = daily_volatility * sqrt(periods_per_year)
    return float(annualized_volatility)


def compare_market(
    snapshot: MarketSnapshot,
    history: list[tuple[date, float]],
    *,
    rate: float,
    dividend_yield: float,
    target_days: int = 30,
) -> dict:
    """Compare an approximately ATM 30-day call with BS at historical vol.

    Rates/yields are explicit user assumptions, not sourced curves. SPXW
    expiry is approximated as 16:00 New York (early closes are not modeled).
    Last 252 available daily returns are used, excluding future closes.
    """
    if not isfinite(rate) or not isfinite(dividend_yield):
        raise ValueError("rate and dividend_yield must be finite")
    if (
        not isinstance(target_days, int)
        or isinstance(target_days, bool)
        or target_days < 1
    ):
        raise ValueError("target_days must be a positive integer")
    as_of = snapshot.as_of.astimezone(NEW_YORK)
    eligible = []
    for option in snapshot.options:
        days_to_expiry = (option.expiry - as_of.date()).days
        if option.option_type == "call" and 7 <= days_to_expiry <= 365:
            eligible.append(option)
    if not eligible:
        raise ValueError("No call with 7 to 365 days to expiry")

    def selection_key(option: OptionQuote) -> tuple[float, float, float]:
        days_to_expiry = (option.expiry - as_of.date()).days
        maturity_distance = abs(days_to_expiry - target_days)
        strike_distance = abs(option.strike - snapshot.spot)
        spread = option.ask - option.bid
        return maturity_distance, strike_distance, spread

    quote = min(eligible, key=selection_key)
    expiry = datetime.combine(quote.expiry, time(16), tzinfo=NEW_YORK)
    time_to_expiry = expiry.astimezone(timezone.utc) - as_of.astimezone(timezone.utc)
    seconds_per_year = 365 * 24 * 60 * 60
    maturity = time_to_expiry.total_seconds() / seconds_per_year

    usable = []
    for day, close in history:
        closing_time = datetime.combine(day, time(16), tzinfo=NEW_YORK)
        if closing_time <= as_of:
            usable.append((day, close))
    usable.sort()
    if len({day for day, _ in usable}) != len(usable):
        raise ValueError("Duplicate history dates")
    if len(usable) < 253:
        raise ValueError("Need 253 daily closes for 252-return historical volatility")
    usable = usable[-253:]
    closes = [close for _, close in usable]
    historical_vol = historical_volatility(closes)
    inputs = dict(
        spot=snapshot.spot,
        strike=quote.strike,
        maturity=maturity,
        rate=rate,
        option_type=quote.option_type,
        dividend_yield=dividend_yield,
    )
    result = implied_volatility(quote.mid, **inputs)
    model = black_scholes_price(**inputs, volatility=historical_vol)
    return {
        "sources": {"options": OPTIONS_URL, "history": HISTORY_URL},
        "retrieved_at": snapshot.retrieved_at,
        "provider_timestamp": snapshot.provider_timestamp,
        "valuation_time": as_of.isoformat(),
        "symbol": quote.symbol,
        "spot": snapshot.spot,
        "strike": quote.strike,
        "expiry": expiry.isoformat(),
        "maturity": maturity,
        "bid": quote.bid,
        "ask": quote.ask,
        "mid": quote.mid,
        "rate_assumption": rate,
        "dividend_yield_assumption": dividend_yield,
        "historical_volatility": historical_vol,
        "implied_volatility": result.volatility,
        "model_price_at_historical_vol": model,
        "model_minus_mid": model - quote.mid,
        "iv_iterations": result.iterations,
        "iv_price_error": result.price_error,
        "history": [{"date": day.isoformat(), "close": close} for day, close in usable],
        "limitations": "Delayed, possibly asynchronous quotes; last underlying trade used as valuation time; "
        "16:00 NY expiry approximation excludes early closes; r/q assumed constant; "
        "historical volatility is not implied volatility; mid is not an executable price.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Assumed continuously compounded rate, decimal",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        required=True,
        help="Assumed continuous dividend yield, decimal",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/market_snapshot.json")
    )
    args = parser.parse_args()
    snapshot = fetch_spx_snapshot()
    history = fetch_spx_history(end_date=snapshot.as_of.astimezone(NEW_YORK).date())
    report = compare_market(
        snapshot, history, rate=args.rate, dividend_yield=args.dividend_yield
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
