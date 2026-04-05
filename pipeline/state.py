"""Pipeline state shared across all LangGraph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts (right overwrites left keys)."""
    merged = {**left}
    merged.update(right)
    return merged


class PipelineState(TypedDict, total=False):
    # Step 1 outputs
    tickers: list[str]
    market_snapshot: dict[str, Any]

    # Step 2 outputs — ticker -> serialised feature DataFrame rows
    ticker_data: Annotated[dict[str, Any], _merge_dicts]

    # Step 2.5 outputs — institutional / smart money
    institutional_signals: Annotated[dict[str, Any], _merge_dicts]

    # Step 3 outputs
    technical_signals: Annotated[dict[str, Any], _merge_dicts]
    pattern_analyses: Annotated[dict[str, Any], _merge_dicts]

    # Step 4 outputs
    decisions: list[dict[str, Any]]

    # Step 5 outputs
    options_recs: Annotated[dict[str, Any], _merge_dicts]

    # Step 6 outputs
    final_report: str

    # Timing metadata
    timing: Annotated[dict[str, float], _merge_dicts]
