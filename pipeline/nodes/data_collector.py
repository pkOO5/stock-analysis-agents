"""Step 2: Data Collector — parallel OHLCV fetch + feature engineering per ticker."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from pipeline.utils import load_config, timed


def _collect_one(ticker: str, lookback_days: int) -> dict[str, Any] | None:
    """Fetch OHLCV + features for a single ticker. Returns serialisable dict."""
    from market_fetcher import fetch_stock_data, fetch_vix
    from feature_engineering import add_features

    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    try:
        df = fetch_stock_data(ticker, start_s, end_s)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        vix = fetch_vix(start_s, end_s)
        df = add_features(df, vix)

        last_20 = df.tail(20).copy()
        last_20.index = last_20.index.strftime("%Y-%m-%d")
        return {
            "ticker": ticker,
            "rows": len(df),
            "last_close": round(float(df["Close"].iloc[-1]), 2),
            "latest": df.iloc[-1].to_dict(),
            "last_20": last_20.to_dict(orient="records"),
        }
    except Exception as e:
        print(f"  [data_collector] {ticker} failed: {e}")
        return None


@timed("data_collector")
def run_data_collector(state: dict[str, Any]) -> dict[str, Any]:
    """Fetch data for all screened tickers in parallel."""
    tickers = state.get("tickers", [])
    cfg = load_config()
    lookback = cfg.get("lookback_days", 1095)

    ticker_data: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_collect_one, t, lookback): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                result = fut.result()
                if result:
                    ticker_data[t] = result
            except Exception as e:
                print(f"  [data_collector] {t} exception: {e}")

    print(f"  [data_collector] collected {len(ticker_data)}/{len(tickers)} tickers")
    return {"ticker_data": ticker_data}
