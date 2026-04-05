"""Step 5: Options Agent — fetch chains via yfinance + LLM strategy recs."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

import yfinance as yf

from pipeline.prompts.templates import OPTIONS_PROMPT, OPTIONS_SYSTEM
from pipeline.utils import ask_llm_json, timed


def _fetch_options_chain(ticker: str) -> dict[str, Any]:
    """Fetch nearest-expiry options chain and compute summary stats."""
    t = yf.Ticker(ticker)
    expirations = t.options
    if not expirations:
        return {"available": False}

    exp = expirations[0]
    chain = t.option_chain(exp)
    calls = chain.calls
    puts = chain.puts

    if calls.empty or puts.empty:
        return {"available": False}

    try:
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    except Exception:
        price = 0

    if price <= 0:
        return {"available": False}

    atm_range = (price * 0.95, price * 1.05)
    atm_calls = calls[(calls["strike"] >= atm_range[0]) & (calls["strike"] <= atm_range[1])]
    atm_puts = puts[(puts["strike"] >= atm_range[0]) & (puts["strike"] <= atm_range[1])]

    iv_col = "impliedVolatility"
    atm_iv = 0.0
    if iv_col in atm_calls.columns and atm_calls[iv_col].notna().any():
        atm_iv = float(atm_calls[iv_col].mean()) * 100

    call_oi = float(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
    put_oi = float(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
    pc_ratio = put_oi / call_oi if call_oi > 0 else 1.0

    def _fmt_chain(df):
        cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].head(8).to_string(index=False)

    return {
        "available": True,
        "expiry": exp,
        "price": price,
        "atm_iv": round(atm_iv, 1),
        "pc_ratio": round(pc_ratio, 2),
        "calls_str": _fmt_chain(atm_calls),
        "puts_str": _fmt_chain(atm_puts),
    }


def _analyze_options(ticker: str, action: str, confidence: float, chain: dict) -> dict[str, Any]:
    """Ask LLM for options strategy recommendation."""
    if not chain.get("available"):
        return {
            "ticker": ticker, "vehicle": "stock",
            "strategy": "direct_stock",
            "reasoning": "Options chain unavailable.",
        }

    prompt = OPTIONS_PROMPT.format(
        ticker=ticker,
        action=action,
        confidence=confidence,
        price=chain["price"],
        expiry=chain["expiry"],
        calls=chain["calls_str"],
        puts=chain["puts_str"],
        atm_iv=chain["atm_iv"],
        pc_ratio=chain["pc_ratio"],
    )

    fallback = {
        "ticker": ticker,
        "vehicle": "options" if chain["atm_iv"] < 40 else "stock",
        "strategy": "long_call" if action == "BUY" else "long_put",
        "strike": round(chain["price"], 0),
        "expiry": chain["expiry"],
        "risk_reward": "Dry-run estimate",
        "reasoning": "Dry-run: basic directional options play.",
    }

    try:
        return ask_llm_json(prompt, system=OPTIONS_SYSTEM, dry_run_response=fallback)
    except Exception as e:
        return {
            "ticker": ticker, "vehicle": "stock",
            "strategy": "direct_stock",
            "reasoning": f"Options agent error: {e}",
        }


@timed("options")
def run_options(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate options strategies for all actionable decisions."""
    decisions = state.get("decisions", [])
    actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]

    recs: dict[str, Any] = {}
    for dec in actionable:
        ticker = dec["ticker"]
        try:
            chain = _fetch_options_chain(ticker)
            recs[ticker] = _analyze_options(
                ticker, dec["action"], dec.get("confidence", 0.5), chain,
            )
        except Exception as e:
            recs[ticker] = {
                "ticker": ticker, "vehicle": "stock",
                "strategy": "direct_stock",
                "reasoning": f"Error: {e}",
            }

    return {"options_recs": recs}
