#!/usr/bin/env python3
"""Soak test: screener tickers (yfinance) + OHLCV (Polygon/Finnhub/Yahoo) + ML path."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

# Shorter lookback for soak (full 1095d + Finnhub chunks is very slow per ticker)
SOAK_LOOKBACK_DAYS = min(500, int(os.environ.get("SOAK_LOOKBACK_DAYS", "500")))
DURATION_SEC = int(os.environ.get("SOAK_DURATION_SEC", "240"))  # default 4 minutes


def main() -> None:
    from pipeline.utils import load_config
    from pipeline.nodes.screener import run_screener
    from pipeline.nodes.data_collector import _collect_one
    from pipeline.nodes.technical import _run_ml_prediction

    cfg = load_config()
    screener_out = run_screener({})
    tickers = screener_out.get("tickers") or []
    if not tickers:
        print("No tickers from screener; exiting.")
        return

    print(f"Soak: {DURATION_SEC}s | lookback={SOAK_LOOKBACK_DAYS}d | tickers={len(tickers)}: {tickers}")
    deadline = time.time() + DURATION_SEC
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        t_iter = time.time()
        for t in tickers:
            row = _collect_one(t, SOAK_LOOKBACK_DAYS)
            if not row:
                print(f"  [it {iteration}] {t}: no data")
                continue
            ml = _run_ml_prediction(row.get("latest", {}), row.get("last_20", []), cfg)
            print(
                f"  [it {iteration}] {t}: signal={ml['signal']} "
                f"conf={ml['confidence']} rows={row.get('rows', '?')}"
            )
        print(f"  --- iteration {iteration} wall {time.time() - t_iter:.1f}s ---")

    print(f"Done: {iteration} full passes over {len(tickers)} tickers.")


if __name__ == "__main__":
    main()
