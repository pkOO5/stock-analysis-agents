#!/usr/bin/env python3
"""
Portfolio simulator — shows dollar outcomes from $10,000 starting capital.

Uses the same walk-forward ML model as the pipeline, but compounds returns
to produce a realistic equity curve with daily/weekly/monthly/yearly breakdowns.

Supports equal-weight across tickers and per-ticker standalone.

Usage:
    python simulate_portfolio.py
    python simulate_portfolio.py --capital 10000 --tickers AAPL NVDA XOM
"""

from __future__ import annotations

import argparse
import os
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


def simulate_ticker(
    df: pd.DataFrame,
    cfg: dict,
    starting_capital: float,
    train_window: int = 252,
    confidence_threshold: float = 0.53,
    max_risk_per_trade: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate compounded returns for a single ticker.

    max_risk_per_trade: fraction of current equity allocated per trade (1.0 = 100%).
    Returns DataFrame indexed by date with columns: equity, daily_return, signal.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    close = df["Close"]
    target = (close.shift(-1) > close).astype(int)
    n = len(df)

    rows = []
    equity = starting_capital

    for i in range(train_window, n - 1):
        train_start = max(0, i - train_window)
        X_train = df[available].iloc[train_start:i].fillna(0)
        y_train = target.iloc[train_start:i]

        if y_train.nunique() < 2 or len(X_train) < 50:
            rows.append({
                "date": df.index[i], "equity": equity,
                "daily_return": 0.0, "signal": "SKIP",
            })
            continue

        model = get_model(cfg)
        weights = get_crisis_weights(df.index[train_start:i], cfg.get("crisis_downweight", 0.3))
        try:
            model.fit(X_train, y_train, sample_weight=weights)
        except Exception:
            rows.append({
                "date": df.index[i], "equity": equity,
                "daily_return": 0.0, "signal": "SKIP",
            })
            continue

        X_test = df[available].iloc[[i]].fillna(0)
        proba = model.predict_proba(X_test)[0]
        pred_up_prob = proba[1] if len(proba) > 1 else proba[0]
        confidence = float(max(proba))

        if confidence >= confidence_threshold:
            signal = "BUY" if pred_up_prob > 0.5 else "SELL"
        else:
            signal = "HOLD"

        actual_return = float((close.iloc[i + 1] - close.iloc[i]) / close.iloc[i])

        if signal == "BUY":
            trade_return = actual_return * max_risk_per_trade
        elif signal == "SELL":
            trade_return = -actual_return * max_risk_per_trade
        else:
            trade_return = 0.0

        equity *= (1 + trade_return)
        rows.append({
            "date": df.index[i],
            "equity": round(equity, 2),
            "daily_return": round(trade_return * 100, 4),
            "signal": signal,
        })

    return pd.DataFrame(rows).set_index("date")


def simulate_portfolio(
    ticker_results: dict[str, pd.DataFrame],
    starting_capital: float,
) -> pd.DataFrame:
    """
    Equal-weight portfolio: split capital across tickers, compound independently,
    then aggregate.
    """
    n_tickers = len(ticker_results)
    if n_tickers == 0:
        return pd.DataFrame()

    per_ticker_capital = starting_capital / n_tickers

    all_dates = set()
    for df in ticker_results.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    rows = []
    for date in all_dates:
        total_equity = 0
        for ticker, df in ticker_results.items():
            if date in df.index:
                total_equity += df.loc[date, "equity"]
            else:
                idx = df.index[df.index <= date]
                if len(idx) > 0:
                    total_equity += df.loc[idx[-1], "equity"]
                else:
                    total_equity += per_ticker_capital

        rows.append({"date": date, "equity": round(total_equity, 2)})

    return pd.DataFrame(rows).set_index("date")


def print_report(
    portfolio_df: pd.DataFrame,
    ticker_dfs: dict[str, pd.DataFrame],
    starting_capital: float,
):
    """Print a formatted report with timeframe breakdowns."""
    print()
    print("=" * 70)
    print(f"  Portfolio Simulation — Starting Capital: ${starting_capital:,.0f}")
    print("=" * 70)
    print()

    # Per-ticker final equity
    print("  Per-Ticker Results (equal-weight allocation)")
    print(f"  {'Ticker':8s} {'Start':>10s} {'End':>10s} {'Return':>10s} {'Trades':>7s}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    n_tickers = len(ticker_dfs)
    per_ticker_start = starting_capital / n_tickers

    for ticker, df in ticker_dfs.items():
        final = df["equity"].iloc[-1]
        ret = (final - per_ticker_start) / per_ticker_start * 100
        trades = len(df[df["signal"].isin(["BUY", "SELL"])])
        print(f"  {ticker:8s} ${per_ticker_start:>9,.0f} ${final:>9,.0f} {ret:>+9.1f}% {trades:>7d}")

    # Portfolio totals
    final_equity = portfolio_df["equity"].iloc[-1]
    total_ret = (final_equity - starting_capital) / starting_capital * 100
    print()
    print(f"  {'TOTAL':8s} ${starting_capital:>9,.0f} ${final_equity:>9,.0f} {total_ret:>+9.1f}%")

    # Time period breakdowns
    total_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    total_trading_days = len(portfolio_df)

    print()
    print("  " + "-" * 50)
    print("  Timeframe Breakdown")
    print("  " + "-" * 50)

    daily_returns = portfolio_df["equity"].pct_change().dropna()
    avg_daily_ret = daily_returns.mean() * 100
    avg_daily_dollar = avg_daily_ret / 100 * starting_capital

    # Per trading day
    print(f"\n  Per Trading Day (avg):")
    print(f"    Return:       {avg_daily_ret:>+.3f}%")
    print(f"    Dollar P&L:   ${avg_daily_dollar:>+,.2f}")

    # Per week (~5 trading days)
    weekly_equity = portfolio_df["equity"].resample("W").last().dropna()
    if len(weekly_equity) > 1:
        weekly_returns = weekly_equity.pct_change().dropna()
        avg_weekly_ret = weekly_returns.mean() * 100
        avg_weekly_dollar = avg_weekly_ret / 100 * starting_capital
        best_week = weekly_returns.max() * 100
        worst_week = weekly_returns.min() * 100
        print(f"\n  Per Week (avg):")
        print(f"    Return:       {avg_weekly_ret:>+.2f}%")
        print(f"    Dollar P&L:   ${avg_weekly_dollar:>+,.2f}")
        print(f"    Best week:    {best_week:>+.2f}%")
        print(f"    Worst week:   {worst_week:>+.2f}%")

    # Per month
    monthly_equity = portfolio_df["equity"].resample("ME").last().dropna()
    if len(monthly_equity) > 1:
        monthly_returns = monthly_equity.pct_change().dropna()
        avg_monthly_ret = monthly_returns.mean() * 100
        avg_monthly_dollar = avg_monthly_ret / 100 * starting_capital
        best_month = monthly_returns.max() * 100
        worst_month = monthly_returns.min() * 100
        winning_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        print(f"\n  Per Month (avg):")
        print(f"    Return:       {avg_monthly_ret:>+.2f}%")
        print(f"    Dollar P&L:   ${avg_monthly_dollar:>+,.2f}")
        print(f"    Best month:   {best_month:>+.2f}%")
        print(f"    Worst month:  {worst_month:>+.2f}%")
        print(f"    Win months:   {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")

    # Per year
    yearly_equity = portfolio_df["equity"].resample("YE").last().dropna()
    if len(yearly_equity) > 1:
        yearly_returns = yearly_equity.pct_change().dropna()
        print(f"\n  Per Year:")
        for date, ret in yearly_returns.items():
            year = date.year
            eq_at_year_end = yearly_equity.loc[date]
            print(f"    {year}:  {ret*100:>+.2f}%  (equity: ${eq_at_year_end:>,.0f})")

    # Max drawdown
    cummax = portfolio_df["equity"].cummax()
    drawdown = (portfolio_df["equity"] - cummax) / cummax * 100
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    print(f"\n  Max Drawdown:   {max_dd:.1f}% (on {max_dd_date.strftime('%Y-%m-%d')})")

    # Projection (if the pattern continued)
    print()
    print("  " + "-" * 50)
    print("  Projection (if this pattern continued)")
    print("  " + "-" * 50)
    annualized_ret = (1 + daily_returns.mean()) ** 252 - 1

    for label, years in [("1 Month", 1 / 12), ("3 Months", 0.25),
                         ("6 Months", 0.5), ("1 Year", 1), ("3 Years", 3)]:
        projected = starting_capital * (1 + annualized_ret) ** years
        print(f"    {label:12s}  ${projected:>10,.0f}  ({(projected/starting_capital - 1)*100:>+.1f}%)")

    print()
    print("  NOTE: Projections assume past patterns repeat — they won't.")
    print("        Past performance does not predict future results.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Portfolio simulation")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital (default: $10,000)")
    parser.add_argument("--tickers", nargs="+", help="Tickers (default: config.yaml)")
    parser.add_argument("--model", choices=["random_forest", "xgboost"])
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--threshold", type=float, default=0.53)
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["model"] = args.model

    tickers = args.tickers or cfg.get("tickers", ["AAPL", "NVDA", "XOM", "BA", "SPY"])
    lookback = cfg.get("lookback_days", 1095)

    print(f"  Simulating ${args.capital:,.0f} across {', '.join(tickers)}...")
    print()

    ticker_dfs = {}
    per_ticker_capital = args.capital / len(tickers)

    for ticker in tickers:
        print(f"  [{ticker}] Fetching data...")
        df = _prepare_data(ticker, lookback)
        if df is None or len(df) < args.train_window + 50:
            print(f"  [{ticker}] Insufficient data, skipping")
            continue

        print(f"  [{ticker}] Simulating on {len(df)} bars...")
        result = simulate_ticker(
            df, cfg, per_ticker_capital,
            train_window=args.train_window,
            confidence_threshold=args.threshold,
        )
        ticker_dfs[ticker] = result

    if not ticker_dfs:
        print("  No tickers simulated.")
        return

    portfolio_df = simulate_portfolio(ticker_dfs, args.capital)
    print_report(portfolio_df, ticker_dfs, args.capital)


if __name__ == "__main__":
    main()
