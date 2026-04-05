#!/usr/bin/env python3
"""
Paper Trade Logger — records pipeline decisions and checks outcomes.

Two modes:
  1. `record`  — called automatically by the pipeline after each run;
                 appends decisions to data/paper_trades.jsonl
  2. `review`  — run manually (or via cron) to check outcomes of past
                 trades after 1, 5, and 10 trading days

Usage:
    python paper_trader.py record '{"decisions": [...], "options_recs": {...}}'
    python paper_trader.py review
    python paper_trader.py review --days 5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


TRADES_FILE = os.path.join(os.path.dirname(__file__), "data", "paper_trades.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)


def record_decisions(
    decisions: list[dict],
    options_recs: dict | None = None,
    market_snapshot: dict | None = None,
) -> int:
    """Append actionable decisions to the paper trades log. Returns count recorded."""
    _ensure_dir()
    actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]
    if not actionable:
        return 0

    timestamp = datetime.now().isoformat()
    recorded = 0

    for dec in actionable:
        ticker = dec["ticker"]

        try:
            t = yf.Ticker(ticker)
            info = t.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        except Exception:
            price = 0

        entry = {
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": dec["action"],
            "confidence": dec.get("confidence", 0),
            "reasoning": dec.get("reasoning", ""),
            "entry_price": round(float(price), 2) if price else None,
            "options_rec": (options_recs or {}).get(ticker, {}),
            "vix": (market_snapshot or {}).get("vix"),
            "outcomes": {},
        }

        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        recorded += 1

    print(f"  [paper_trader] recorded {recorded} trades to {TRADES_FILE}")
    return recorded


def _load_trades() -> list[dict]:
    """Load all paper trades from the JSONL file."""
    if not os.path.exists(TRADES_FILE):
        return []
    trades = []
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return trades


def _save_trades(trades: list[dict]):
    """Overwrite the JSONL file with updated trades."""
    _ensure_dir()
    with open(TRADES_FILE, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def _get_price_on_date(ticker: str, target_date: str) -> float | None:
    """Fetch closing price on or near a specific date."""
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        closest = df.index[df.index <= pd.Timestamp(target_date)]
        if closest.empty:
            closest = df.index
        return float(df.loc[closest[-1], "Close"])
    except Exception:
        return None


def review_outcomes(check_days: list[int] | None = None) -> list[dict]:
    """Check outcomes for all recorded trades. Updates the JSONL file in place."""
    check_days = check_days or [1, 5, 10]
    trades = _load_trades()
    if not trades:
        print("  [paper_trader] No trades recorded yet.")
        return []

    updated = False
    today = datetime.now()

    for trade in trades:
        entry_date = datetime.strptime(trade["date"], "%Y-%m-%d")
        entry_price = trade.get("entry_price")
        if not entry_price:
            continue

        outcomes = trade.get("outcomes", {})

        for days in check_days:
            key = f"day_{days}"
            if key in outcomes:
                continue

            check_date = entry_date + timedelta(days=days)
            if check_date > today:
                continue

            exit_price = _get_price_on_date(
                trade["ticker"],
                check_date.strftime("%Y-%m-%d"),
            )
            if exit_price is None:
                continue

            pct_change = (exit_price - entry_price) / entry_price * 100
            trade_return = pct_change if trade["action"] == "BUY" else -pct_change

            outcomes[key] = {
                "exit_price": round(exit_price, 2),
                "pct_change": round(pct_change, 2),
                "trade_return_pct": round(trade_return, 2),
                "correct": trade_return > 0,
            }
            updated = True

        trade["outcomes"] = outcomes

    if updated:
        _save_trades(trades)

    return trades


def print_review(trades: list[dict], check_days: list[int] | None = None):
    """Print a formatted review of paper trade outcomes."""
    check_days = check_days or [1, 5, 10]

    if not trades:
        print("No trades to review.")
        return

    has_outcomes = [t for t in trades if t.get("outcomes")]
    pending = [t for t in trades if not t.get("outcomes")]

    print()
    print("=" * 80)
    print("  Paper Trade Review")
    print("=" * 80)
    print()

    for d in check_days:
        key = f"day_{d}"
        relevant = [t for t in has_outcomes if key in t.get("outcomes", {})]
        if not relevant:
            continue

        wins = sum(1 for t in relevant if t["outcomes"][key]["correct"])
        total = len(relevant)
        avg_ret = sum(t["outcomes"][key]["trade_return_pct"] for t in relevant) / total

        print(f"  {d}-Day Outcomes ({total} trades)")
        print(f"  {'Ticker':8s} {'Action':6s} {'Entry':>8s} {'Exit':>8s} {'Return':>8s} {'Result':>8s}")
        print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for t in relevant:
            o = t["outcomes"][key]
            result = "WIN" if o["correct"] else "LOSS"
            print(f"  {t['ticker']:8s} {t['action']:6s} "
                  f"${t['entry_price']:>7.2f} ${o['exit_price']:>7.2f} "
                  f"{o['trade_return_pct']:>+7.2f}% {result:>8s}")

        print(f"\n  Win rate: {wins}/{total} ({wins/total*100:.0f}%)  |  "
              f"Avg return: {avg_ret:+.2f}%\n")

    if pending:
        print(f"  {len(pending)} trade(s) still pending outcome checks.\n")


def main():
    parser = argparse.ArgumentParser(description="Paper trade logger")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="Record decisions from pipeline output")
    rec.add_argument("json_data", help="JSON string with decisions and options_recs")

    rev = sub.add_parser("review", help="Review outcomes of past trades")
    rev.add_argument("--days", type=int, nargs="+", default=[1, 5, 10],
                     help="Check outcome after N days (default: 1 5 10)")

    args = parser.parse_args()

    if args.command == "record":
        data = json.loads(args.json_data)
        record_decisions(
            data.get("decisions", []),
            data.get("options_recs"),
            data.get("market_snapshot"),
        )
    elif args.command == "review":
        trades = review_outcomes(args.days)
        print_review(trades, args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
