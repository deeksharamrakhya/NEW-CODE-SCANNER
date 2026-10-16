"""
analysis.py
Stock analysis module: fetches prices (via the cached data layer) + runs the
pattern detectors for any ticker.

Works for both Indian (NSE: 'RELIANCE.NS', BSE: 'RELIANCE.BO') and
US tickers (e.g. 'AAPL').
"""

from patterns import fetch_ohlcv, PATTERN_REGISTRY

try:
    from data import get_prices, clamp_period
except Exception:  # data.py optional: fall back to direct downloads
    get_prices = None

    def clamp_period(period, interval):
        return period


def _fmt_pct(x):
    return f"{x:.2f}%" if x is not None else "N/A"


def get_fundamentals(ticker: str) -> dict:
    """Pull basic fundamental snapshot. Falls back gracefully if fields are missing."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "currency": info.get("currency"),
    }


def analyze_stock(ticker: str, period: str = "1y", include_fundamentals: bool = True,
                  patterns: list = None, interval: str = "1d",
                  detector_kwargs: dict = None) -> dict:
    """
    Full analysis for a single ticker.

    patterns        : pattern names to run (default: all in PATTERN_REGISTRY).
    interval        : bar size ('1d','1wk','1h','15m', ...). `period` is auto-clamped
                      to what yfinance allows for intraday intervals.
    detector_kwargs : {pattern_name: {kwarg: value}}, e.g. patterns.PRESETS["Short-term (2-4 wk)"].
    """
    used_period = clamp_period(period, interval)
    if get_prices is not None:
        df = get_prices(ticker, period=used_period, interval=interval)   # cached
    else:
        df = fetch_ohlcv(ticker, period=used_period, interval=interval)

    dk = detector_kwargs or {}
    names_to_run = patterns if patterns else list(PATTERN_REGISTRY.keys())
    pattern_results = {}
    for name in names_to_run:
        detector_fn = PATTERN_REGISTRY.get(name)
        if detector_fn is None:
            continue
        try:
            pattern_results[name] = detector_fn(df, **dk.get(name, {}))
        except Exception as e:   # one broken detector shouldn't kill the whole ticker
            pattern_results[name] = {"pattern": name, "detected": False, "error": str(e)}

    result = {
        "ticker": ticker,
        "as_of": str(df.index[-1].date()),
        "last_close": round(float(df["close"].iloc[-1]), 2),
        "period_used": used_period,
        "interval": interval,
        "patterns": pattern_results,
    }
    result["patterns_detected"] = [
        p["pattern"] for p in result["patterns"].values() if p.get("detected")
    ]

    if include_fundamentals:
        try:
            result["fundamentals"] = get_fundamentals(ticker)
        except Exception as e:
            result["fundamentals"] = {"error": str(e)}
    return result


def summarize(result: dict) -> str:
    """Plain-English one-block summary of an analyze_stock() result."""
    lines = []
    f = result.get("fundamentals", {})
    name = f.get("name", result["ticker"])
    lines.append(f"{name} ({result['ticker']}) — as of {result['as_of']}, last close {result['last_close']}")

    if f and "error" not in f:
        pe = f.get("pe_ratio")
        lines.append(
            f"  Sector: {f.get('sector', 'N/A')} | PE: {round(pe,1) if pe else 'N/A'} | "
            f"52w range: {f.get('52w_low','N/A')}–{f.get('52w_high','N/A')}"
        )

    detected = result["patterns_detected"]
    if detected:
        lines.append(f"  Patterns flagged: {', '.join(detected)}")
        for p in result["patterns"].values():
            if p.get("detected"):
                details = {k: v for k, v in p.items() if k not in ("pattern", "detected")}
                lines.append(f"    - {p['pattern']}: {details}")
    else:
        lines.append("  No patterns flagged — price action doesn't match any of the tracked setups right now.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RELIANCE.NS", "AAPL"]
    for t in tickers:
        try:
            print("\n" + summarize(analyze_stock(t)))
        except Exception as e:
            print(f"\n{t}: ERROR — {e}")
