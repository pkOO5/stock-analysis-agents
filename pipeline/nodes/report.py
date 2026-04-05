"""Step 6: Report Agent — synthesise all outputs into a final markdown brief."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipeline.prompts.templates import REPORT_PROMPT, REPORT_SYSTEM
from pipeline.utils import ask_llm, timed


def _decisions_summary(decisions: list[dict]) -> str:
    lines = []
    for d in decisions:
        lines.append(
            f"- {d['ticker']}: {d.get('action', 'HOLD')} "
            f"(confidence={d.get('confidence', '?')}) — {d.get('reasoning', '')}"
        )
    return "\n".join(lines) or "None"


def _options_summary(recs: dict) -> str:
    if not recs:
        return "No options recommendations (no actionable signals or chains unavailable)."
    lines = []
    for ticker, rec in recs.items():
        lines.append(
            f"- {ticker}: vehicle={rec.get('vehicle', '?')}, "
            f"strategy={rec.get('strategy', '?')}, "
            f"strike={rec.get('strike', 'N/A')}, expiry={rec.get('expiry', 'N/A')}\n"
            f"  Reasoning: {rec.get('reasoning', '')}"
        )
    return "\n".join(lines)


@timed("report")
def run_report(state: dict[str, Any]) -> dict[str, Any]:
    """Generate the final executive report."""
    snapshot = state.get("market_snapshot", {})
    tickers = state.get("tickers", [])
    decisions = state.get("decisions", [])
    options_recs = state.get("options_recs", {})
    timing = state.get("timing", {})

    market_overview = (
        f"VIX: {snapshot.get('vix', '?')} ({snapshot.get('vix_regime', '?')})\n"
        f"Screener reasoning: {snapshot.get('reasoning', 'N/A')}"
    )

    prompt = REPORT_PROMPT.format(
        market_overview=market_overview,
        tickers=", ".join(tickers),
        decisions=_decisions_summary(decisions),
        options_recs=_options_summary(options_recs),
        timing=json.dumps(timing, indent=2),
    )

    dry_run_report = (
        f"# Pipeline Report (Dry Run)\n\n"
        f"## Market Summary\n{market_overview}\n\n"
        f"## Screened Tickers\n{', '.join(tickers)}\n\n"
        f"## Decisions\n{_decisions_summary(decisions)}\n\n"
        f"## Options Recommendations\n{_options_summary(options_recs)}\n\n"
        f"## Timing\n{json.dumps(timing, indent=2)}\n\n"
        f"*Note: This is a dry-run report (Ollama not running). "
        f"Start Ollama for full LLM-powered analysis.*\n"
    )

    report_md = ask_llm(
        prompt, system=REPORT_SYSTEM, tier="fast",
        dry_run_response=dry_run_report,
    )

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = os.path.join(data_dir, f"pipeline_report_{date_str}.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"  [report] written to {report_path}")

    # Auto-record decisions for paper trading
    try:
        from paper_trader import record_decisions
        record_decisions(decisions, options_recs, snapshot)
    except Exception as e:
        print(f"  [report] paper_trader record failed: {e}")

    return {"final_report": report_md}
