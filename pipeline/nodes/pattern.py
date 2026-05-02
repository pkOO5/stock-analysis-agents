"""Step 3b: Pattern Analysis Agent — candlestick + price action via LLM."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

from pipeline.prompts.templates import PATTERN_PROMPT, PATTERN_SYSTEM
from pipeline.utils import ask_llm_json, timed


def _pattern_abbrev(bar: dict) -> str:
    """Compact candle flags: D=doji H=hammer I=inverted B=bull engulf E=bear engulf."""
    parts = []
    if int(bar.get("cdl_doji", 0) or 0):
        parts.append("D")
    if int(bar.get("cdl_hammer", 0) or 0):
        parts.append("H")
    if int(bar.get("cdl_inverted_hammer", 0) or 0):
        parts.append("I")
    if int(bar.get("cdl_bullish_engulfing", 0) or 0):
        parts.append("B")
    if int(bar.get("cdl_bearish_engulfing", 0) or 0):
        parts.append("E")
    return "".join(parts) or "-"


def _format_bars(last_20: list[dict]) -> str:
    """Format last 20 bars as OHLCV + algorithmic candlestick flags (see CANDLESTICK_PATTERNS.md)."""
    lines = [
        "Date       | Open     | High     | Low      | Close    | Volume   | Cdl",
        "-" * 85,
    ]
    for bar in last_20:
        lines.append(
            f"{str(bar.get('date', '?')):10s} | "
            f"{bar.get('Open', 0):8.2f} | {bar.get('High', 0):8.2f} | "
            f"{bar.get('Low', 0):8.2f} | {bar.get('Close', 0):8.2f} | "
            f"{bar.get('Volume', 0):8.0f} | {_pattern_abbrev(bar)}"
        )
    return "\n".join(lines)


def analyze_patterns(ticker: str, ticker_data: dict) -> dict[str, Any]:
    """Run LLM pattern analysis for a single ticker."""
    last_20 = ticker_data.get("last_20", [])
    latest = ticker_data.get("latest", {})

    bars_str = _format_bars(last_20)

    prompt = PATTERN_PROMPT.format(
        ticker=ticker,
        bars=bars_str,
        doji=int(latest.get("cdl_doji", 0)),
        hammer=int(latest.get("cdl_hammer", 0)),
        inv_hammer=int(latest.get("cdl_inverted_hammer", 0)),
        bull_engulf=int(latest.get("cdl_bullish_engulfing", 0)),
        bear_engulf=int(latest.get("cdl_bearish_engulfing", 0)),
        hammer_5d=int(latest.get("cdl_hammer_5d", 0)),
        doji_5d=int(latest.get("cdl_doji_5d", 0)),
    )

    fallback = {
        "ticker": ticker,
        "trend": "neutral",
        "patterns": [],
        "confidence": 0.5,
        "reasoning": "Dry-run: no pattern analysis available.",
    }

    try:
        result = ask_llm_json(prompt, system=PATTERN_SYSTEM, tier="fast", dry_run_response=fallback)
    except Exception as e:
        result = {
            "ticker": ticker,
            "trend": "neutral",
            "patterns": [],
            "confidence": 0.5,
            "reasoning": f"LLM error: {e}",
        }

    result["ticker"] = ticker
    return result


@timed("pattern")
def run_pattern(state: dict[str, Any]) -> dict[str, Any]:
    """Run pattern analysis across all tickers."""
    ticker_data = state.get("ticker_data", {})
    analyses: dict[str, Any] = {}

    for ticker, data in ticker_data.items():
        try:
            analyses[ticker] = analyze_patterns(ticker, data)
        except Exception as e:
            analyses[ticker] = {
                "ticker": ticker, "trend": "neutral",
                "patterns": [], "confidence": 0.5,
                "reasoning": f"Error: {e}",
            }

    return {"pattern_analyses": analyses}
