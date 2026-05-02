#!/usr/bin/env python3
"""
Live (latest-bar) ML check using the same stack as the pipeline technical agent.

Fetches current OHLCV via market_fetcher (Polygon → Finnhub → yfinance), builds
features including candlesticks, fits on all completed bars except using the
last row only as X for prediction — same logic as pipeline/nodes/technical.py.

With --report, appends a walk-forward evaluation on the same fetched history
(risk/reward, drawdown, weekday-of-signal patterns, candlestick buckets). Daily
data has no true clock-time-of-day; weekday is a calendar proxy only.

Usage:
    python live_ml_signal.py
    python live_ml_signal.py --tickers AAPL NVDA MSFT
    python live_ml_signal.py --model xgboost
    python live_ml_signal.py --report --report-train-window 252
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from constants import FEATURE_COLS, get_crisis_weights, get_model
from feature_engineering import add_features
from market_fetcher import fetch_stock_data, fetch_vix

from backtest import summarise, walk_forward_backtest

_WDAY = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _row_for_date(df: pd.DataFrame, date_str: str) -> pd.Series | None:
    try:
        ts = pd.Timestamp(date_str).normalize()
        for i, ix in enumerate(df.index):
            if pd.Timestamp(ix).normalize() == ts:
                return df.iloc[i]
    except Exception:
        pass
    return None


def _enrich_trade_records(df: pd.DataFrame, records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        row = _row_for_date(df, r["date"])
        if row is None:
            r2 = dict(r)
            r2["weekday"] = None
            r2["cdl_tag"] = "?"
            out.append(r2)
            continue
        dow = int(row["day_of_week"]) if "day_of_week" in row.index and pd.notna(row["day_of_week"]) else None
        tags = []
        for col, name in [
            ("cdl_hammer", "hammer"),
            ("cdl_inverted_hammer", "inv_hammer"),
            ("cdl_doji", "doji"),
            ("cdl_bullish_engulfing", "bull_eng"),
            ("cdl_bearish_engulfing", "bear_eng"),
        ]:
            if col in row.index and int(row[col]) == 1:
                tags.append(name)
        r2 = dict(r)
        r2["weekday"] = dow
        r2["cdl_tag"] = "+".join(tags) if tags else "plain"
        out.append(r2)
    return out


def _max_drawdown_pct(cumulative_returns: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    run = 0.0
    for x in cumulative_returns:
        run += x
        peak = max(peak, run)
        max_dd = min(max_dd, run - peak)
    return round(max_dd, 3)


def _print_performance_report(
    df: pd.DataFrame,
    cfg: dict,
    ticker: str,
    train_window: int,
    threshold: float,
) -> None:
    records = walk_forward_backtest(
        df, cfg, train_window=train_window, step=1, confidence_threshold=threshold
    )
    enriched = _enrich_trade_records(df, records)
    trades = [r for r in enriched if r["signal"] != "HOLD"]

    print()
    print(f"  === Walk-forward report: {ticker}  (train_window={train_window}d, threshold={threshold}) ===")
    if not trades:
        print("  No trades above threshold in this window.")
        return

    summary = summarise(records, ticker)
    rets = [r["trade_return_pct"] for r in trades]
    wins = [r for r in trades if r["correct"]]
    losses = [r for r in trades if not r["correct"]]
    sum_win = sum(r["trade_return_pct"] for r in wins)
    sum_loss = sum(r["trade_return_pct"] for r in losses)
    avg_win = sum_win / len(wins) if wins else 0.0
    avg_loss = sum_loss / len(losses) if losses else 0.0
    profit_factor = (
        sum_win / abs(sum_loss) if sum_loss < 0 else float("inf") if sum_win > 0 else 0.0
    )
    reward_risk = (
        abs(avg_win / avg_loss) if losses and avg_loss != 0 else float("inf") if avg_win > 0 else 0.0
    )
    std_ret = float(np.std(rets)) if len(rets) > 1 else 0.0
    eq = list(np.cumsum(rets))
    mdd = _max_drawdown_pct(rets)

    streak_loss = cur = 0
    for r in trades:
        if not r["correct"]:
            cur += 1
            streak_loss = max(streak_loss, cur)
        else:
            cur = 0

    print(
        f"  Trades: {summary['trades']}  |  Win rate: {summary['win_rate_pct']}%  "
        f"|  Total P/L: {summary['total_return_pct']:+.2f}%  |  Avg/trade: {summary['avg_return_per_trade_pct']:+.3f}%"
    )
    print(
        f"  Best: {summary['best_trade_pct']:+.2f}%  Worst: {summary['worst_trade_pct']:+.2f}%  "
        f"|  Stdev(trade): {std_ret:.3f}%  |  Max drawdown (cum %): {mdd}%"
    )
    pf_s = f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf"
    rr_s = f"{reward_risk:.2f}" if reward_risk != float("inf") else "inf"
    print(
        f"  Profit factor (gross win / gross loss): {pf_s}  |  "
        f"Avg win {avg_win:+.3f}% / Avg loss {avg_loss:+.3f}%  |  |avgWin|/|avgLoss|: {rr_s}"
    )
    print(f"  Max losing streak (trades): {streak_loss}  |  BUY {summary['buys']} / SELL {summary['sells']}")

    print()
    print("  --- Time context (daily bars = EOD only; no intraday clock time) ---")
    print("  Weekday of **signal bar** (0=Mon … 4=Fri; proxy for calendar seasonality):")

    by_dow: dict[int, list[dict]] = {}
    for r in trades:
        d = r.get("weekday")
        if d is None or d > 4:
            continue
        by_dow.setdefault(d, []).append(r)
    for d in range(5):
        lst = by_dow.get(d, [])
        if not lst:
            print(f"    {_WDAY[d]}: (no trades)")
            continue
        wr = 100.0 * sum(1 for x in lst if x["correct"]) / len(lst)
        tot = sum(x["trade_return_pct"] for x in lst)
        print(f"    {_WDAY[d]}: n={len(lst):3d}  win%={wr:5.1f}  sum_ret={tot:+.2f}%")

    print()
    print("  --- Candlestick tag on signal bar (algorithmic; see CANDLESTICK_PATTERNS.md) ---")
    by_tag: dict[str, list[dict]] = {}
    for r in trades:
        by_tag.setdefault(r.get("cdl_tag", "?"), []).append(r)
    for tag in sorted(by_tag.keys(), key=lambda t: -len(by_tag[t])):
        lst = by_tag[tag]
        wr = 100.0 * sum(1 for x in lst if x["correct"]) / len(lst)
        tot = sum(x["trade_return_pct"] for x in lst)
        print(f"    {tag:16s} n={len(lst):3d}  win%={wr:5.1f}  sum_ret={tot:+.2f}%")

    print("  === end report ===")


def _load_config() -> dict:
    path = os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _prepare(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    df = fetch_stock_data(ticker, start_s, end_s)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    vix = fetch_vix(start_s, end_s)
    df = add_features(df, vix)
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 5:
        return None
    return df.dropna(subset=available + ["Close"])


def _predict_latest(df: pd.DataFrame, cfg: dict, min_train: int) -> dict | None:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    close = df["Close"]
    y = (close.shift(-1) > close).astype(int)
    y_train = y.iloc[:-1]
    X_train = X.iloc[:-1]
    if len(X_train) < min_train or y_train.nunique() < 2:
        return None
    model = get_model(cfg)
    w = get_crisis_weights(df.index[:-1], cfg.get("crisis_downweight", 0.3))
    try:
        model.fit(X_train, y_train, sample_weight=w)
    except TypeError:
        model.fit(X_train, y_train)
    proba = model.predict_proba(X.iloc[[-1]])[0]
    conf = float(max(proba))
    pred_up = float(proba[1] if len(proba) > 1 else proba[0])
    sig = "BUY" if np.argmax(proba) == 1 else "SELL"
    if conf < 0.53:
        sig = "HOLD"
    row = df.iloc[-1]
    cdl = {k: int(row[k]) for k in available if k.startswith("cdl_")}
    as_of = df.index[-1]
    as_of_s = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)[:10]
    return {
        "ticker": "",
        "as_of": as_of_s,
        "last_close": round(float(close.iloc[-1]), 4),
        "signal": sig,
        "confidence": round(conf, 4),
        "p_next_close_up": round(pred_up, 4),
        "candlestick": cdl,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ML + candlestick signal on latest fetched bar")
    ap.add_argument("--tickers", nargs="+", help="Symbols (default: config.yaml tickers)")
    ap.add_argument("--lookback-days", type=int, help="History window (default: config lookback_days)")
    ap.add_argument("--min-train-rows", type=int, default=80, help="Minimum in-sample rows before predict")
    ap.add_argument(
        "--model",
        choices=["random_forest", "xgboost"],
        help="Override config.yaml model",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="After live line, run walk-forward on same data (risk, reward, weekday/candle patterns)",
    )
    ap.add_argument(
        "--report-train-window",
        type=int,
        default=252,
        help="Rolling train window for --report (default: 252)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.53,
        help="ML confidence gate for trades in --report (default: 0.53, same as technical.py)",
    )
    args = ap.parse_args()
    cfg = _load_config()
    if args.model:
        cfg["model"] = args.model
    tickers = args.tickers or cfg.get("tickers", ["SPY", "AAPL"])
    lookback = args.lookback_days or int(cfg.get("lookback_days", 1095))

    print("Live ML signal (latest completed daily bar → P(next close up))")
    print(f"  Model: {cfg.get('model', 'random_forest')}  |  lookback: {lookback}d")
    print("-" * 72)

    for t in tickers:
        df = _prepare(t, lookback)
        if df is None or len(df) < args.min_train_rows + 2:
            print(f"  {t:6s}  SKIP  (no data or rows<{args.min_train_rows + 2})")
            continue
        out = _predict_latest(df, cfg, args.min_train_rows)
        if not out:
            print(f"  {t:6s}  SKIP  (need varied labels + min_train)")
            continue
        out["ticker"] = t
        cdl_s = " ".join(f"{k}={v}" for k, v in sorted(out["candlestick"].items())) or "(no cdl cols)"
        print(
            f"  {t:6s}  as_of={out['as_of']}  close={out['last_close']:>10.2f}  "
            f"signal={out['signal']:<4s}  conf={out['confidence']:.3f}  P(up)={out['p_next_close_up']:.3f}"
        )
        print(f"           {cdl_s}")
        if args.report:
            if len(df) < args.report_train_window + 50:
                print(f"  {t:6s}  (skip report: need >= {args.report_train_window + 50} rows)")
            else:
                _print_performance_report(
                    df, cfg, t, args.report_train_window, args.threshold,
                )
    print("-" * 72)
    print(
        "Note: Daily OHLCV only — no tick-level time-of-day. "
        "Use --report for P/L stats; weekday = calendar day of the signal bar."
    )


if __name__ == "__main__":
    main()
