#!/usr/bin/env python3
"""CLI entry point for the multi-agent stock analysis pipeline.

Usage:
    python -m pipeline.run_pipeline
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def main():
    from pipeline.graph import build_graph
    from pipeline.utils import load_config

    cfg = load_config()
    timeout = cfg.get("pipeline", {}).get("timeout_minutes", 15) * 60

    print("=" * 60)
    print("  Multi-Agent Stock Analysis Pipeline")
    print("=" * 60)
    print()

    app = build_graph()
    t0 = time.time()

    initial_state = {
        "tickers": [],
        "market_snapshot": {},
        "ticker_data": {},
        "institutional_signals": {},
        "technical_signals": {},
        "pattern_analyses": {},
        "decisions": [],
        "options_recs": {},
        "final_report": "",
        "timing": {},
    }

    final_state = app.invoke(initial_state, {"recursion_limit": 50})

    total = round(time.time() - t0, 1)
    print()
    print("=" * 60)
    print(f"  Pipeline complete in {total}s")
    print("=" * 60)

    timing = final_state.get("timing", {})
    for step, secs in timing.items():
        print(f"    {step:20s}  {secs:6.1f}s")
    print(f"    {'TOTAL':20s}  {total:6.1f}s")
    print()

    decisions = final_state.get("decisions", [])
    actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]
    print(f"  Tickers screened:  {len(final_state.get('tickers', []))}")
    print(f"  Data collected:    {len(final_state.get('ticker_data', {}))}")
    print(f"  Actionable calls:  {len(actionable)}")
    print()

    if actionable:
        print("  Action Summary:")
        for d in actionable:
            options_rec = final_state.get("options_recs", {}).get(d["ticker"], {})
            vehicle = options_rec.get("vehicle", "stock")
            strategy = options_rec.get("strategy", "direct")
            print(
                f"    {d['ticker']:6s}  {d['action']:4s}  "
                f"conf={d.get('confidence', '?')}  "
                f"vehicle={vehicle}  strategy={strategy}"
            )
        print()

    if total > timeout:
        print(f"  WARNING: exceeded {timeout}s timeout")


if __name__ == "__main__":
    main()
