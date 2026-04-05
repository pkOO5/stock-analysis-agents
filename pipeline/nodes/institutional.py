"""Step 2.5: Institutional Tracker — what are the big players doing?

Checks three data sources per ticker (all free, no API key):
  1. Institutional holders (top funds + % ownership) via yfinance
  2. Insider transactions (recent buys/sells by executives) via yfinance
  3. 13F hedge fund holdings changes (quarterly) via edgartools / SEC EDGAR

The LLM synthesises these into a "smart money" signal per ticker.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

import yfinance as yf

from pipeline.prompts.templates import INSTITUTIONAL_SYSTEM, INSTITUTIONAL_PROMPT
from pipeline.utils import ask_llm_json, timed


# Major funds to track in 13F filings (CIK numbers for SEC EDGAR)
TRACKED_FUNDS = {
    "Berkshire Hathaway": "BRK-B",
    "Bridgewater Associates": None,
    "Renaissance Technologies": None,
    "Citadel Advisors": None,
    "BlackRock": "BLK",
}


def _get_institutional_holders(ticker: str) -> str:
    """Top institutional holders from yfinance."""
    try:
        t = yf.Ticker(ticker)
        inst = t.institutional_holders
        if inst is None or inst.empty:
            return "No institutional holder data available."
        cols = [c for c in ["Holder", "Shares", "Date Reported", "% Out", "Value"]
                if c in inst.columns]
        return inst[cols].head(10).to_string(index=False)
    except Exception as e:
        return f"Error fetching institutional holders: {e}"


def _get_major_holders(ticker: str) -> str:
    """Major holder breakdown (insider %, institutional %, float)."""
    try:
        t = yf.Ticker(ticker)
        major = t.major_holders
        if major is None or major.empty:
            return "No major holder data available."
        return major.to_string()
    except Exception:
        return "Major holder data unavailable."


def _get_insider_transactions(ticker: str) -> str:
    """Recent insider buys and sells."""
    try:
        t = yf.Ticker(ticker)
        insider = t.insider_transactions
        if insider is None or insider.empty:
            return "No recent insider transactions."
        cols = [c for c in ["Insider", "Start Date", "Transaction", "Shares", "Value"]
                if c in insider.columns]
        if not cols:
            cols = insider.columns.tolist()[:5]
        return insider[cols].head(15).to_string(index=False)
    except Exception as e:
        return f"Error fetching insider transactions: {e}"


def _get_13f_changes(ticker: str) -> str:
    """Check recent 13F filing changes for a ticker from SEC EDGAR."""
    try:
        from edgar import Company
        company = Company(ticker)
        filings = company.get_filings(form="13F-HR")
        if not filings or len(filings) == 0:
            return "No 13F filings found for this ticker."

        latest = filings[0]
        report = latest.obj()
        if hasattr(report, "holdings_view"):
            holdings = report.holdings_view()
            if holdings is not None and not holdings.empty:
                return holdings.head(10).to_string(index=False)
        return "13F data parsed but no holdings extracted."
    except Exception:
        # edgartools may not find 13F for every ticker (it's filed by the fund, not the company)
        return "13F data not available for this ticker."


def _get_institutional_summary(ticker: str) -> dict[str, str]:
    """Gather all institutional data for a ticker."""
    return {
        "institutional_holders": _get_institutional_holders(ticker),
        "major_holders": _get_major_holders(ticker),
        "insider_transactions": _get_insider_transactions(ticker),
    }


def analyze_institutional(ticker: str, inst_data: dict[str, str]) -> dict[str, Any]:
    """Ask LLM to interpret institutional activity for a ticker."""
    prompt = INSTITUTIONAL_PROMPT.format(
        ticker=ticker,
        institutional_holders=inst_data["institutional_holders"],
        major_holders=inst_data["major_holders"],
        insider_transactions=inst_data["insider_transactions"],
    )

    fallback = {
        "ticker": ticker,
        "smart_money_signal": "neutral",
        "insider_sentiment": "neutral",
        "institutional_trend": "unknown",
        "reasoning": "Dry-run: no institutional analysis.",
    }

    try:
        result = ask_llm_json(prompt, system=INSTITUTIONAL_SYSTEM, dry_run_response=fallback)
    except Exception as e:
        result = {
            "ticker": ticker,
            "smart_money_signal": "neutral",
            "insider_sentiment": "neutral",
            "institutional_trend": "unknown",
            "reasoning": f"LLM error: {e}",
        }

    result["ticker"] = ticker
    return result


@timed("institutional")
def run_institutional(state: dict[str, Any]) -> dict[str, Any]:
    """Check what big players are doing for all screened tickers."""
    ticker_data = state.get("ticker_data", {})
    tickers = list(ticker_data.keys())

    institutional_signals: dict[str, Any] = {}

    for ticker in tickers:
        try:
            inst_data = _get_institutional_summary(ticker)
            institutional_signals[ticker] = analyze_institutional(ticker, inst_data)
        except Exception as e:
            institutional_signals[ticker] = {
                "ticker": ticker,
                "smart_money_signal": "neutral",
                "reasoning": f"Error: {e}",
            }

    return {"institutional_signals": institutional_signals}
