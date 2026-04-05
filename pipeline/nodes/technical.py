"""Step 3a: Technical Analysis Agent — ML predictions + LLM reasoning."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

import numpy as np
import pandas as pd

from constants import FEATURE_COLS, get_model
from pipeline.prompts.templates import TECHNICAL_PROMPT, TECHNICAL_SYSTEM
from pipeline.utils import ask_llm_json, load_config, timed


def _run_ml_prediction(df_latest: dict, last_20: list[dict], cfg: dict) -> dict[str, Any]:
    """Train model on last_20 rows and predict the latest bar direction."""
    df = pd.DataFrame(last_20)

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 5 or len(df) < 10:
        return {"signal": "HOLD", "confidence": 0.5, "importances": {}}

    X = df[available].fillna(0)
    close = df.get("Close")
    if close is None or len(close) < 2:
        return {"signal": "HOLD", "confidence": 0.5, "importances": {}}

    y = (close.shift(-1) > close).astype(int)
    y = y.iloc[:-1]
    X_train = X.iloc[:-1]

    if len(X_train) < 5 or y.nunique() < 2:
        return {"signal": "HOLD", "confidence": 0.5, "importances": {}}

    model = get_model(cfg)
    try:
        model.fit(X_train, y)
    except Exception:
        return {"signal": "HOLD", "confidence": 0.5, "importances": {}}

    X_latest = X.iloc[[-1]]
    proba = model.predict_proba(X_latest)[0]
    confidence = float(max(proba))
    signal = "BUY" if np.argmax(proba) == 1 else "SELL"
    if confidence < 0.53:
        signal = "HOLD"

    importances = {}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        pairs = sorted(zip(available, fi), key=lambda x: x[1], reverse=True)
        importances = {k: round(float(v), 3) for k, v in pairs[:5]}

    return {"signal": signal, "confidence": round(confidence, 3), "importances": importances}


def analyze_one_ticker(ticker: str, ticker_data: dict, cfg: dict) -> dict[str, Any]:
    """Run ML + LLM technical analysis for a single ticker."""
    last_20 = ticker_data.get("last_20", [])
    latest = ticker_data.get("latest", {})

    ml_result = _run_ml_prediction(latest, last_20, cfg)

    indicators_lines = []
    for key in ["RSI", "EMA_20", "EMA_50", "MACD", "MACD_signal", "ATR",
                 "bb_bandwidth", "bb_position", "volatility", "volume_ma_ratio", "VIX"]:
        val = latest.get(key)
        if val is not None:
            indicators_lines.append(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    price_lines = []
    for bar in last_20[-10:]:
        line = (
            f"  {bar.get('date', '?')} O={bar.get('Open', '?'):.2f} "
            f"H={bar.get('High', '?'):.2f} L={bar.get('Low', '?'):.2f} "
            f"C={bar.get('Close', '?'):.2f} V={bar.get('Volume', 0):.0f}"
        )
        price_lines.append(line)

    prompt = TECHNICAL_PROMPT.format(
        ticker=ticker,
        signal=ml_result["signal"],
        confidence=ml_result["confidence"],
        importances=ml_result["importances"],
        indicators="\n".join(indicators_lines) or "N/A",
        price_action="\n".join(price_lines) or "N/A",
    )

    fallback = {
        "ticker": ticker,
        "signal": ml_result["signal"],
        "ml_confidence": ml_result["confidence"],
        "adjusted_confidence": ml_result["confidence"],
        "reasoning": "Dry-run: using raw ML output.",
        "key_levels": {},
    }

    try:
        result = ask_llm_json(prompt, system=TECHNICAL_SYSTEM, tier="fast", dry_run_response=fallback)
    except Exception as e:
        result = {
            "ticker": ticker,
            "signal": ml_result["signal"],
            "ml_confidence": ml_result["confidence"],
            "adjusted_confidence": ml_result["confidence"],
            "reasoning": f"LLM error: {e}",
            "key_levels": {},
        }

    result["ml_result"] = ml_result
    return result


@timed("technical")
def run_technical(state: dict[str, Any]) -> dict[str, Any]:
    """Run technical analysis across all tickers (sequential — each calls LLM)."""
    cfg = load_config()
    ticker_data = state.get("ticker_data", {})
    signals: dict[str, Any] = {}

    for ticker, data in ticker_data.items():
        try:
            signals[ticker] = analyze_one_ticker(ticker, data, cfg)
        except Exception as e:
            signals[ticker] = {
                "ticker": ticker, "signal": "HOLD",
                "adjusted_confidence": 0.5, "reasoning": f"Error: {e}",
            }

    return {"technical_signals": signals}
