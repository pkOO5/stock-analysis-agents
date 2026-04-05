#!/usr/bin/env python3
"""
Pipeline scheduler — runs the analysis pipeline at fixed intervals
during US stock market hours, Monday through Friday.

Usage:
    python scheduler.py                        # default: every 60 min, market hours
    python scheduler.py --interval 30          # every 30 minutes
    python scheduler.py --interval 10          # every 10 minutes
    python scheduler.py --start 08:30 --end 15:00  # custom window (your local time)
    python scheduler.py --once                 # single run, then exit

The scheduler:
  - Only runs Mon-Fri
  - Only runs between --start and --end (default 8:30-15:00 CT / adjust to your TZ)
  - Sleeps between runs
  - Logs everything to data/scheduler.log
  - Skips a run if the previous one is still going
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import signal
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = os.path.join("data", "scheduler.log")
LOCK = threading.Lock()
_running = True


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs("data", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _is_market_day() -> bool:
    """Monday=0 ... Friday=4."""
    return datetime.now().weekday() < 5


def _in_window(start_time: str, end_time: str) -> bool:
    now = datetime.now().strftime("%H:%M")
    return start_time <= now <= end_time


def _run_pipeline() -> bool:
    """Run the pipeline once. Returns True on success."""
    if not LOCK.acquire(blocking=False):
        _log("SKIP — previous run still in progress")
        return False

    try:
        _log("=== Pipeline run starting ===")
        t0 = time.time()

        from pipeline.graph import build_graph

        app = build_graph()
        initial_state = {
            "tickers": [], "market_snapshot": {}, "ticker_data": {},
            "technical_signals": {}, "pattern_analyses": {},
            "decisions": [], "options_recs": {},
            "final_report": "", "timing": {},
        }

        final_state = app.invoke(initial_state, {"recursion_limit": 50})

        elapsed = round(time.time() - t0, 1)
        decisions = final_state.get("decisions", [])
        actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]

        _log(f"=== Pipeline done in {elapsed}s | "
             f"{len(final_state.get('tickers', []))} tickers | "
             f"{len(actionable)} actionable ===")

        for d in actionable:
            orec = final_state.get("options_recs", {}).get(d["ticker"], {})
            _log(f"  {d['ticker']} {d['action']} conf={d.get('confidence', '?')} "
                 f"vehicle={orec.get('vehicle', 'stock')} "
                 f"strategy={orec.get('strategy', 'direct')}")

        return True

    except Exception as e:
        _log(f"ERROR: {e}")
        return False
    finally:
        LOCK.release()


def _handle_signal(signum, frame):
    global _running
    _log("Received shutdown signal, exiting after current run...")
    _running = False


def main():
    parser = argparse.ArgumentParser(description="Pipeline scheduler")
    parser.add_argument("--interval", type=int, default=60,
                        help="Minutes between runs (default: 60)")
    parser.add_argument("--start", default="08:30",
                        help="Start time HH:MM in your local timezone (default: 08:30)")
    parser.add_argument("--end", default="15:00",
                        help="End time HH:MM in your local timezone (default: 15:00)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (ignore schedule)")
    parser.add_argument("--include-weekends", action="store_true",
                        help="Also run on Saturday/Sunday")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    from dotenv import load_dotenv
    load_dotenv(".env")

    _log(f"Scheduler started | interval={args.interval}min | "
         f"window={args.start}-{args.end} | weekdays={'+ weekends' if args.include_weekends else 'only'}")

    if args.once:
        _run_pipeline()
        return

    while _running:
        now = datetime.now()
        is_weekday = now.weekday() < 5

        if (is_weekday or args.include_weekends) and _in_window(args.start, args.end):
            _run_pipeline()

            if _running:
                next_run = now + timedelta(minutes=args.interval)
                _log(f"Next run at {next_run.strftime('%H:%M')}")
                sleep_secs = args.interval * 60
                for _ in range(sleep_secs):
                    if not _running:
                        break
                    time.sleep(1)
        else:
            if not is_weekday and not args.include_weekends:
                reason = "weekend"
            else:
                reason = f"outside window ({args.start}-{args.end})"
            _log(f"Idle — {reason}. Checking again in 5 min.")
            for _ in range(300):
                if not _running:
                    break
                time.sleep(1)

    _log("Scheduler stopped.")


if __name__ == "__main__":
    main()
