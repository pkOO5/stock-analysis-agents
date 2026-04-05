"""Step 4: Decision Agent — synthesises technical + pattern into BUY/SELL/HOLD."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

from pipeline.prompts.templates import DECISION_PROMPT, DECISION_SYSTEM
from pipeline.utils import ask_llm_json, load_config, timed


def _format_analyses(technical: dict, patterns: dict, institutional: dict) -> str:
    """Build a per-ticker summary block for the decision prompt."""
    tickers = set(list(technical.keys()) + list(patterns.keys()))
    blocks = []
    for t in sorted(tickers):
        tech = technical.get(t, {})
        pat = patterns.get(t, {})
        inst = institutional.get(t, {})
        block = (
            f"### {t}\n"
            f"Technical: signal={tech.get('signal', 'N/A')}, "
            f"confidence={tech.get('adjusted_confidence', tech.get('ml_confidence', 'N/A'))}\n"
            f"  Reasoning: {tech.get('reasoning', 'N/A')}\n"
            f"  Key levels: {tech.get('key_levels', {})}\n"
            f"Pattern: trend={pat.get('trend', 'N/A')}, "
            f"patterns={pat.get('patterns', [])}, confidence={pat.get('confidence', 'N/A')}\n"
            f"  Reasoning: {pat.get('reasoning', 'N/A')}\n"
            f"  Support={pat.get('support', 'N/A')}, Resistance={pat.get('resistance', 'N/A')}\n"
            f"Smart Money: signal={inst.get('smart_money_signal', 'N/A')}, "
            f"insider={inst.get('insider_sentiment', 'N/A')}, "
            f"institutional_trend={inst.get('institutional_trend', 'N/A')}\n"
            f"  {inst.get('reasoning', 'N/A')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


@timed("decision")
def run_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Produce final BUY/SELL/HOLD decisions per ticker."""
    cfg = load_config()
    max_positions = cfg.get("pipeline", {}).get("max_positions", 5)
    snapshot = state.get("market_snapshot", {})
    vix = snapshot.get("vix", 20.0)
    vix_regime = snapshot.get("vix_regime", "moderate")

    technical = state.get("technical_signals", {})
    patterns = state.get("pattern_analyses", {})
    institutional = state.get("institutional_signals", {})

    analyses_str = _format_analyses(technical, patterns, institutional)
    prompt = DECISION_PROMPT.format(
        vix=vix,
        vix_regime=vix_regime,
        analyses=analyses_str,
        max_positions=max_positions,
    )

    fallback_decisions = []
    for t, tech in technical.items():
        sig = tech.get("signal", "HOLD")
        conf = tech.get("adjusted_confidence", tech.get("ml_confidence", 0.5))
        action = sig if sig in ("BUY", "SELL") and conf >= 0.55 else "HOLD"
        fallback_decisions.append({
            "ticker": t, "action": action,
            "confidence": conf, "reasoning": "Dry-run: based on raw ML signal.",
        })
    actionable_count = sum(1 for d in fallback_decisions if d["action"] in ("BUY", "SELL"))
    if actionable_count > max_positions:
        sorted_fb = sorted(fallback_decisions, key=lambda d: d["confidence"], reverse=True)
        kept = 0
        for d in sorted_fb:
            if d["action"] in ("BUY", "SELL"):
                if kept >= max_positions:
                    d["action"] = "HOLD"
                kept += 1

    try:
        result = ask_llm_json(
            prompt, system=DECISION_SYSTEM,
            dry_run_response={"decisions": fallback_decisions},
        )
        decisions = result.get("decisions", [])
    except Exception as e:
        decisions = [
            {"ticker": t, "action": "HOLD", "confidence": 0.5,
             "reasoning": f"Decision agent error: {e}"}
            for t in technical.keys()
        ]

    return {"decisions": decisions}
