"""
backtest.py
Walk-forward backtest for the pattern detectors in patterns.PATTERN_REGISTRY.

WHY WALK-FORWARD:
The detectors in patterns.py only ever judge the MOST RECENT window of the
DataFrame you hand them (they operate on df.tail(lookback)). So to ask
"did this pattern have any edge historically?" we replay history bar by bar:
at each date t we feed the detector df.iloc[:t+1] (exactly what a trader knew
AT THAT TIME -- no lookahead), record a signal when it fires, enter on the next
bar's close and exit `horizon` bars later.

This is a SIGNAL-QUALITY backtest, not a portfolio simulator: fixed holding
period, one unit per signal, no sizing / stops / slippage (a flat cost knob).

Direction:
  bullish patterns -> LONG, bearish patterns -> SHORT. Returns are PnL in the
  direction of the trade, so positive mean_return == "the signal worked".

BASELINE: every result also reports the average return of entering on EVERY
tested bar with the same direction/horizon. `edge_pct = mean_return - baseline`
is the number that says whether the pattern beat "just being long (or short)
this stock", which is the honest comparison.

Detector settings: pass `detector_kwargs` (e.g. patterns.PRESETS[...][name]) so
the backtest tests the SAME pattern definition the live scan used.
"""

import numpy as np
import pandas as pd

from patterns import PATTERN_REGISTRY, fetch_ohlcv, min_bars_required

PATTERN_DIRECTION = {
    "Consolidation": "long",          # neutral by nature; treated as long-bias breakout
    "VCP": "long",
    "Rocket Base": "long",
    "Cup and Handle": "long",
    "Double Top": "short",
    "Double Bottom": "long",
    "Head and Shoulders": "short",
    "Inverse Head and Shoulders": "long",
    "Ascending Triangle": "long",
    "Descending Triangle": "short",
}


def _get_df(ticker: str, period: str) -> pd.DataFrame:
    """Cached daily prices if data.py is available, else a direct fetch."""
    try:
        from data import get_prices
        return get_prices(ticker, period=period)
    except Exception:
        return fetch_ohlcv(ticker, period=period)


def _forward_return(closes: np.ndarray, entry_idx: int, horizon: int) -> float:
    exit_idx = min(entry_idx + horizon, len(closes) - 1)
    return (closes[exit_idx] - closes[entry_idx]) / closes[entry_idx]


def _run(df, pattern_name, horizon, min_history, cooldown, cost_pct, step, detector_kwargs):
    """Core replay. Returns (trades, baseline_returns_pct, detector_errors, tested_bars)."""
    if pattern_name not in PATTERN_REGISTRY:
        raise ValueError(f"Unknown pattern '{pattern_name}'. Options: {list(PATTERN_REGISTRY)}")

    kw = detector_kwargs or {}
    detector = PATTERN_REGISTRY[pattern_name]
    direction = PATTERN_DIRECTION.get(pattern_name, "long")
    sign = 1.0 if direction == "long" else -1.0

    if min_history is None:
        min_history = max(min_bars_required(pattern_name, kw) + 10, 40)
    if cooldown is None:
        # never let trades overlap: the equity maths assumes they are sequential
        cooldown = max(20, horizon)

    closes = df["close"].values
    dates = df.index
    n = len(df)
    last_testable = n - horizon - 1

    trades, errors = [], 0
    last_signal_i = -10 ** 9

    for i in range(min_history, last_testable, step):
        if i - last_signal_i < cooldown:
            continue
        window = df.iloc[: i + 1]                 # information available AS OF bar i
        try:
            res = detector(window, **kw)
        except Exception:
            errors += 1                           # counted, not hidden
            continue
        if not res.get("detected"):
            continue

        entry_idx = i + 1                         # act on the NEXT bar's close
        raw_ret = _forward_return(closes, entry_idx, horizon)
        trade_ret = sign * raw_ret - cost_pct / 100.0
        trades.append({
            "signal_date": str(dates[i].date()),
            "entry_date": str(dates[entry_idx].date()),
            "entry_price": round(float(closes[entry_idx]), 2),
            "direction": direction,
            "raw_forward_return_pct": round(float(raw_ret) * 100, 2),
            "trade_return_pct": round(float(trade_ret) * 100, 2),
        })
        last_signal_i = i

    # Baseline: same direction/horizon/cost, entered on every tested bar.
    entries = np.arange(min_history + 1, last_testable + 1, step)
    if len(entries):
        exits = np.minimum(entries + horizon, n - 1)
        base = (sign * (closes[exits] / closes[entries] - 1.0) * 100.0) - cost_pct
    else:
        base = np.array([])
    tested = len(range(min_history, max(last_testable, min_history), step))
    return trades, base, errors, tested


def _aggregate(pattern_name, direction, horizon, trades, baseline, errors, tested, pooled=False):
    base_mean = round(float(baseline.mean()), 2) if len(baseline) else None
    out = {
        "pattern": pattern_name, "direction": direction, "horizon_bars": horizon,
        "n_trades": len(trades), "detector_errors": errors, "bars_tested": tested,
        "baseline_mean_pct": base_mean,
    }
    if errors and tested and errors >= 0.5 * tested:
        out["warning"] = (f"Detector raised an error on {errors}/{tested} windows - "
                          f"zero signals here may be a bug, not a quiet pattern.")
    if not trades:
        out.update(note="No signals fired over this history window.", trades=[])
        return out

    rets = np.array([t["trade_return_pct"] for t in trades])
    std = float(rets.std())
    out.update({
        "win_rate_pct": round(float((rets > 0).mean() * 100), 1),
        "mean_return_pct": round(float(rets.mean()), 2),
        "median_return_pct": round(float(np.median(rets)), 2),
        "edge_pct": round(float(rets.mean()) - base_mean, 2) if base_mean is not None else None,
        "best_pct": round(float(rets.max()), 2),
        "worst_pct": round(float(rets.min()), 2),
        "std_pct": round(std, 2),
        # per-trade mean/std (NOT annualised) - only for comparing patterns to each other
        "sharpe_like": round(float(rets.mean()) / std, 2) if std > 0 else 0.0,
    })
    if pooled:
        out["max_drawdown_pct"] = None   # a single equity curve across tickers is meaningless
        out["final_equity_x"] = None
    else:
        equity = np.cumprod(1 + rets / 100.0)
        peak = np.maximum.accumulate(equity)
        out["max_drawdown_pct"] = round(float(((equity - peak) / peak).min() * 100), 2)
        out["final_equity_x"] = round(float(equity[-1]), 3)
    out["trades"] = trades
    return out


def backtest_pattern(df: pd.DataFrame, pattern_name: str, horizon: int = 20,
                     min_history: int = None, cooldown: int = None,
                     cost_pct: float = 0.0, step: int = 1,
                     detector_kwargs: dict = None) -> dict:
    """
    Replay `df` bar by bar and evaluate one pattern.

    horizon         : bars held after entry (entry = next bar's close).
    min_history     : warm-up bars before testing. Default: what the detector
                      needs (+10), so short-window presets don't waste data.
    cooldown        : bars to skip after a signal. Default max(20, horizon) so
                      trades never overlap.
    cost_pct        : round-trip friction in percent (0.2 = 20 bps).
    step            : evaluate every `step` bars (raise to speed up).
    detector_kwargs : kwargs for THIS pattern's detector (e.g. from PRESETS).
    """
    direction = PATTERN_DIRECTION.get(pattern_name, "long")
    trades, base, errors, tested = _run(df, pattern_name, horizon, min_history,
                                        cooldown, cost_pct, step, detector_kwargs)
    return _aggregate(pattern_name, direction, horizon, trades, base, errors, tested)


def backtest_pooled(tickers: list, pattern_name: str, period: str = "5y",
                    horizon: int = 20, detector_kwargs: dict = None,
                    cost_pct: float = 0.0, step: int = 1, min_history: int = None,
                    cooldown: int = None, progress=None) -> dict:
    """
    Run one pattern across many tickers and pool the trades. This is how you get
    a sample size worth trusting (one stock gives a handful of signals).
    Caveats: survivorship bias if `tickers` is today's index list, and trades on
    different tickers can overlap in calendar time (so no pooled equity curve).
    """
    direction = PATTERN_DIRECTION.get(pattern_name, "long")
    all_trades, bases, errors, tested = [], [], 0, 0
    used, with_signals, failed = 0, 0, []

    for k, tk in enumerate(tickers):
        try:
            df = _get_df(tk, period)
            tr, base, er, te = _run(df, pattern_name, horizon, min_history,
                                    cooldown, cost_pct, step, detector_kwargs)
        except Exception as e:
            failed.append((tk, str(e)))
            continue
        used += 1
        errors += er
        tested += te
        bases.append(base)
        if tr:
            with_signals += 1
            for t in tr:
                t["ticker"] = tk
            all_trades.extend(tr)
        if progress:
            progress((k + 1) / len(tickers))

    all_trades.sort(key=lambda t: t["entry_date"])
    baseline = np.concatenate(bases) if bases else np.array([])
    out = _aggregate(pattern_name, direction, horizon, all_trades, baseline,
                     errors, tested, pooled=True)
    out.update(n_tickers=used, tickers_with_signals=with_signals, failed=failed)
    return out


def backtest_ticker(ticker: str, pattern_name: str, period: str = "5y",
                    horizon: int = 20, **kwargs) -> dict:
    """Fetch prices for a ticker, then backtest one pattern on it."""
    df = _get_df(ticker, period)
    out = backtest_pattern(df, pattern_name, horizon=horizon, **kwargs)
    out["ticker"] = ticker
    out["period"] = period
    return out


def backtest_all_patterns(ticker: str, period: str = "5y", horizon: int = 20,
                          detector_kwargs: dict = None, **kwargs) -> pd.DataFrame:
    """Run every registered pattern on one ticker. detector_kwargs = {pattern: {...}}."""
    df = _get_df(ticker, period)
    dk = detector_kwargs or {}
    rows = []
    for name in PATTERN_REGISTRY:
        r = backtest_pattern(df, name, horizon=horizon, detector_kwargs=dk.get(name), **kwargs)
        rows.append({
            "pattern": name, "direction": r["direction"], "n_trades": r["n_trades"],
            "win_rate_pct": r.get("win_rate_pct"), "mean_return_pct": r.get("mean_return_pct"),
            "baseline_pct": r.get("baseline_mean_pct"), "edge_pct": r.get("edge_pct"),
            "sharpe_like": r.get("sharpe_like"), "max_dd_pct": r.get("max_drawdown_pct"),
            "final_equity_x": r.get("final_equity_x"),
        })
    return pd.DataFrame(rows).sort_values(
        "edge_pct", ascending=False, na_position="last").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from patterns import PRESETS
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "Consolidation"
    preset = sys.argv[3] if len(sys.argv) > 3 else "Default"
    kw = PRESETS[preset].get(pattern)
    print(f"Backtesting '{pattern}' on {ticker} (preset: {preset}) ...")
    r = backtest_ticker(ticker, pattern, period="5y", horizon=10, detector_kwargs=kw)
    for k, v in r.items():
        if k != "trades":
            print(f"  {k}: {v}")
    print(f"  first 3 trades: {r['trades'][:3]}")
