#!/usr/bin/env python3
"""
Walk-forward backtest for the ML signal model.

Trains on a rolling window, predicts the next day, tracks accuracy and
simulated returns. This validates whether the ML layer (the foundation
of the pipeline) has any real predictive edge.

Usage:
    python backtest.py                      # all tickers in config
    python backtest.py --tickers AAPL NVDA  # specific tickers
    python backtest.py --model xgboost      # override model
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yaml

from constants import FEATURE_COLS, get_model, get_crisis_weights
from feature_engineering import add_features
from market_fetcher import fetch_stock_data, fetch_vix


def load_config() -> dict:
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        return yaml.safe_load(f) or {}


def _prepare_data(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    """Fetch and feature-engineer data for a ticker."""
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

    df = df.dropna(subset=available + ["Close"])
    return df


def walk_forward_backtest(
    df: pd.DataFrame,
    cfg: dict,
    train_window: int = 252,
    step: int = 1,
    confidence_threshold: float = 0.53,
) -> list[dict[str, Any]]:
    """
    Walk-forward backtest:
      - Train on [i - train_window : i]
      - Predict day i
      - Record result
      - Slide forward by `step` days

    Returns list of per-prediction records.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    close = df["Close"]
    target = (close.shift(-1) > close).astype(int)

    records = []
    n = len(df)

    for i in range(train_window, n - 1, step):
        train_start = max(0, i - train_window)
        X_train = df[available].iloc[train_start:i].fillna(0)
        y_train = target.iloc[train_start:i]

        if y_train.nunique() < 2 or len(X_train) < 50:
            continue

        model = get_model(cfg)
        weights = get_crisis_weights(
            df.index[train_start:i],
            cfg.get("crisis_downweight", 0.3),
        )

        try:
            model.fit(X_train, y_train, sample_weight=weights)
        except Exception:
            continue

        X_test = df[available].iloc[[i]].fillna(0)
        proba = model.predict_proba(X_test)[0]
        pred_up_prob = proba[1] if len(proba) > 1 else proba[0]
        confidence = float(max(proba))

        if confidence >= confidence_threshold:
            signal = "BUY" if pred_up_prob > 0.5 else "SELL"
        else:
            signal = "HOLD"

        actual_return = float(
            (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i]
        )
        actual_up = actual_return > 0

        correct = (
            (signal == "BUY" and actual_up)
            or (signal == "SELL" and not actual_up)
        )

        records.append({
            "date": str(df.index[i].date()),
            "signal": signal,
            "confidence": round(confidence, 3),
            "actual_return_pct": round(actual_return * 100, 3),
            "actual_up": actual_up,
            "correct": correct if signal != "HOLD" else None,
            "trade_return_pct": round(
                actual_return * 100 * (1 if signal == "BUY" else -1), 3
            ) if signal != "HOLD" else 0.0,
        })

    return records


def summarise(records: list[dict], ticker: str) -> dict[str, Any]:
    """Compute aggregate stats from backtest records."""
    trades = [r for r in records if r["signal"] != "HOLD"]
    holds = [r for r in records if r["signal"] == "HOLD"]

    if not trades:
        return {"ticker": ticker, "trades": 0, "note": "No signals above threshold"}

    wins = [r for r in trades if r["correct"]]
    win_rate = len(wins) / len(trades) * 100
    total_return = sum(r["trade_return_pct"] for r in trades)
    avg_return = total_return / len(trades)
    buys = [r for r in trades if r["signal"] == "BUY"]
    sells = [r for r in trades if r["signal"] == "SELL"]

    return {
        "ticker": ticker,
        "total_predictions": len(records),
        "trades": len(trades),
        "holds": len(holds),
        "wins": len(wins),
        "losses": len(trades) - len(wins),
        "win_rate_pct": round(win_rate, 1),
        "total_return_pct": round(total_return, 2),
        "avg_return_per_trade_pct": round(avg_return, 3),
        "buys": len(buys),
        "sells": len(sells),
        "best_trade_pct": round(max(r["trade_return_pct"] for r in trades), 2),
        "worst_trade_pct": round(min(r["trade_return_pct"] for r in trades), 2),
        "avg_confidence": round(
            sum(r["confidence"] for r in trades) / len(trades), 3
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward ML backtest")
    parser.add_argument("--tickers", nargs="+", help="Tickers to test (default: config.yaml)")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], help="Override model")
    parser.add_argument("--train-window", type=int, default=252, help="Training window in days (default: 252)")
    parser.add_argument("--threshold", type=float, default=0.53, help="Confidence threshold (default: 0.53)")
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["model"] = args.model

    tickers = args.tickers or cfg.get("tickers", ["AAPL", "NVDA", "XOM", "BA", "SPY"])
    lookback = cfg.get("lookback_days", 1095)

    print("=" * 65)
    print("  Walk-Forward ML Backtest")
    print(f"  Model: {cfg.get('model', 'random_forest')}  |  "
          f"Train window: {args.train_window}d  |  Threshold: {args.threshold}")
    print(f"  Tickers: {', '.join(tickers)}")
    print("=" * 65)
    print()

    all_summaries = []

    for ticker in tickers:
        print(f"  [{ticker}] Fetching data...")
        df = _prepare_data(ticker, lookback)
        if df is None or len(df) < args.train_window + 50:
            print(f"  [{ticker}] Insufficient data ({len(df) if df is not None else 0} rows), skipping")
            continue

        print(f"  [{ticker}] Running backtest on {len(df)} bars...")
        records = walk_forward_backtest(
            df, cfg,
            train_window=args.train_window,
            confidence_threshold=args.threshold,
        )
        summary = summarise(records, ticker)
        all_summaries.append(summary)

        if summary.get("trades", 0) > 0:
            print(f"  [{ticker}] {summary['trades']} trades | "
                  f"Win rate: {summary['win_rate_pct']}% | "
                  f"Total return: {summary['total_return_pct']:+.2f}% | "
                  f"Avg/trade: {summary['avg_return_per_trade_pct']:+.3f}%")
        else:
            print(f"  [{ticker}] {summary.get('note', 'No trades')}")
        print()

    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    print()
    print(f"  {'Ticker':8s} {'Trades':>7s} {'Win%':>6s} {'TotalRet':>10s} {'Avg/Trade':>10s} {'Best':>7s} {'Worst':>7s}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*7}")

    for s in all_summaries:
        if s.get("trades", 0) > 0:
            print(f"  {s['ticker']:8s} {s['trades']:7d} {s['win_rate_pct']:5.1f}% "
                  f"{s['total_return_pct']:+9.2f}% {s['avg_return_per_trade_pct']:+9.3f}% "
                  f"{s['best_trade_pct']:+6.2f}% {s['worst_trade_pct']:+6.2f}%")
        else:
            print(f"  {s['ticker']:8s}       0   N/A        N/A        N/A     N/A     N/A")

    if all_summaries:
        traded = [s for s in all_summaries if s.get("trades", 0) > 0]
        if traded:
            avg_wr = sum(s["win_rate_pct"] for s in traded) / len(traded)
            total_ret = sum(s["total_return_pct"] for s in traded)
            print()
            print(f"  Portfolio: avg win rate {avg_wr:.1f}% | combined return {total_ret:+.2f}%")
    print()


if __name__ == "__main__":
    main()
