"""Step 1: Market Screener Agent — scans broad market, picks 10-15 tickers."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from pipeline.prompts.templates import SCREENER_PROMPT, SCREENER_SYSTEM
from pipeline.utils import ask_llm_json, load_config, timed

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC",
    "NFLX", "CRM", "ORCL", "ADBE", "AVGO", "QCOM",
    "XOM", "CVX", "COP", "OXY",
    "JPM", "BAC", "GS", "MS", "WFC",
    "UNH", "JNJ", "PFE", "ABBV", "LLY",
    "BA", "CAT", "HON", "GE", "RTX",
    "COST", "WMT", "HD", "MCD", "NKE",
    "DIS", "CMCSA", "T", "VZ",
    "F", "GM", "AAL", "DAL",
    "SPY", "QQQ",
]


def _fetch_market_snapshot() -> tuple[pd.DataFrame, float]:
    """Fetch 5-day data for watchlist and current VIX."""
    end = datetime.now()
    start = end - timedelta(days=10)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    try:
        data = yf.download(
            WATCHLIST, start=start_s, end=end_s, progress=False, group_by="ticker"
        )
    except Exception:
        data = pd.DataFrame()

    rows = []
    for ticker in WATCHLIST:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[ticker].dropna(subset=["Close"])
            else:
                df_t = data.dropna(subset=["Close"])
            if df_t.empty or len(df_t) < 2:
                continue
            close = df_t["Close"]
            vol = df_t["Volume"]
            ret_5d = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            vol_ratio = vol.iloc[-1] / vol.mean() if vol.mean() > 0 else 1.0
            rows.append({
                "ticker": ticker,
                "price": round(float(close.iloc[-1]), 2),
                "ret_5d_pct": round(float(ret_5d), 2),
                "vol_ratio": round(float(vol_ratio), 2),
            })
        except Exception:
            continue

    snapshot_df = pd.DataFrame(rows)

    try:
        vix_data = yf.download("^VIX", start=start_s, end=end_s, progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.droplevel(1)
        vix_level = float(vix_data["Close"].iloc[-1])
    except Exception:
        vix_level = 20.0

    return snapshot_df, vix_level


def _vix_regime(vix: float) -> str:
    if vix < 20:
        return "low"
    if vix <= 30:
        return "moderate"
    return "high"


@timed("screener")
def run_screener(state: dict[str, Any]) -> dict[str, Any]:
    """Screen the market and select tickers for deep analysis."""
    cfg = load_config()
    max_tickers = cfg.get("pipeline", {}).get("max_tickers", 15)

    snapshot_df, vix_level = _fetch_market_snapshot()

    if snapshot_df.empty:
        tickers = cfg.get("tickers", ["AAPL", "NVDA", "XOM", "BA", "SPY"])
        return {
            "tickers": tickers[:max_tickers],
            "market_snapshot": {"vix": vix_level, "vix_regime": _vix_regime(vix_level), "data": []},
        }

    market_data_str = snapshot_df.to_string(index=False)
    prompt = SCREENER_PROMPT.format(
        market_data=market_data_str,
        vix_level=vix_level,
        vix_regime=_vix_regime(vix_level),
    )

    sorted_snap = snapshot_df.reindex(
        snapshot_df["ret_5d_pct"].abs().sort_values(ascending=False).index
    )
    fallback_tickers = sorted_snap["ticker"].tolist()[:max_tickers]

    result = ask_llm_json(
        prompt,
        system=SCREENER_SYSTEM,
        dry_run_response={
            "tickers": fallback_tickers,
            "reasoning": "Dry-run: selected top movers by absolute 5d return.",
        },
    )

    valid_symbols = set(snapshot_df["ticker"].tolist())
    tickers = [t for t in result.get("tickers", []) if t in valid_symbols][:max_tickers]

    if not tickers:
        tickers = cfg.get("tickers", ["AAPL", "NVDA", "XOM", "BA", "SPY"])

    return {
        "tickers": tickers,
        "market_snapshot": {
            "vix": vix_level,
            "vix_regime": _vix_regime(vix_level),
            "reasoning": result.get("reasoning", ""),
            "data": snapshot_df.to_dict(orient="records"),
        },
    }
