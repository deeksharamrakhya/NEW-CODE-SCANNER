"""
indices.py
Fetches index constituent tickers for Nifty 50 and S&P 500, formatted for
yfinance (Nifty 50 tickers get the '.NS' suffix automatically).

IMPORTANT: Index composition changes (reconstitutions, M&A, demergers). This
module fetches the CURRENT list live; if that fails it falls back to a small
dated hardcoded list. Pass `with_source=True` to learn which one you got, so the
UI can warn instead of silently scanning a stale / truncated list:

    tickers, source = get_index_constituents("sp500", with_source=True)
    # source is "live" or "fallback"

Verify against the official sources before relying on any fallback:
  - Nifty 50: https://www.nseindia.com/index-nifty50
  - S&P 500 : https://www.spglobal.com/spdji/en/indices/equity/sp-500/
"""

import io
import pandas as pd

# Browser-like header: Wikipedia and NSE both reject the default python UA (HTTP 403).
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Fallback only -- NOT guaranteed current. Last manually checked: 2026-07.
_NIFTY50_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "LT", "KOTAKBANK", "SBIN", "AXISBANK", "HINDUNILVR", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "ADANIENT", "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM", "INDUSINDBK", "BAJAJFINSV",
    "DIVISLAB", "DRREDDY", "CIPLA", "GRASIM", "BRITANNIA", "EICHERMOT",
    "HEROMOTOCO", "BAJAJ-AUTO", "COALINDIA", "HINDALCO", "APOLLOHOSP",
    "SBILIFE", "HDFCLIFE", "BPCL", "SHRIRAMFIN", "UPL", "LTIM", "ADANIPORTS",
]

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX",
]


def _http_text(url: str, timeout: int = 15) -> str:
    import requests
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def get_nifty50_constituents(prefer_live: bool = True, with_source: bool = False):
    """Nifty 50 tickers with '.NS' suffix, ready for yfinance."""
    tickers, source = None, "fallback"
    if prefer_live:
        try:
            text = _http_text("https://archive.nseindia.com/content/indices/ind_nifty50list.csv")
            df = pd.read_csv(io.StringIO(text))
            tickers = df["Symbol"].str.strip().tolist()
            if len(tickers) >= 45:          # sanity check: a real Nifty 50 list
                source = "live"
            else:
                tickers = None
        except Exception:
            tickers = None

    if not tickers:
        print("Note: live Nifty 50 fetch failed -- using fallback list "
              "(may be stale, verify against nseindia.com).")
        tickers = _NIFTY50_FALLBACK

    out = [f"{t}.NS" if not t.endswith(".NS") else t for t in tickers]
    return (out, source) if with_source else out


def get_sp500_constituents(prefer_live: bool = True, with_source: bool = False):
    """S&P 500 tickers, ready for yfinance."""
    tickers, source = None, "fallback"
    if prefer_live:
        try:
            html = _http_text("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            tables = pd.read_html(io.StringIO(html))
            tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            if len(tickers) >= 400:
                source = "live"
            else:
                tickers = None
        except Exception:
            tickers = None

    if not tickers:
        print("Note: live S&P 500 fetch failed -- using a SHORT fallback list "
              "(top names only, NOT the full 500; verify against spglobal.com).")
        tickers = _SP500_FALLBACK

    return (tickers, source) if with_source else tickers


def get_index_constituents(index_name: str, with_source: bool = False):
    index_name = index_name.lower().strip()
    if index_name in ("nifty50", "nifty 50", "nifty"):
        return get_nifty50_constituents(with_source=with_source)
    if index_name in ("sp500", "s&p500", "s&p 500", "spx"):
        return get_sp500_constituents(with_source=with_source)
    raise ValueError(f"Unknown index '{index_name}'. Supported: 'nifty50', 'sp500'")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "nifty50"
    tickers, src = get_index_constituents(name, with_source=True)
    print(f"{name}: {len(tickers)} tickers (source: {src})")
    print(tickers)
