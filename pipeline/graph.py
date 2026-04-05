"""LangGraph StateGraph wiring — orchestrates the 7-step pipeline."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes.screener import run_screener
from pipeline.nodes.data_collector import run_data_collector
from pipeline.nodes.institutional import run_institutional
from pipeline.nodes.technical import run_technical
from pipeline.nodes.pattern import run_pattern
from pipeline.nodes.decision import run_decision
from pipeline.nodes.options import run_options
from pipeline.nodes.report import run_report


def _should_run_options(state: PipelineState) -> str:
    """Skip options step if there are no actionable decisions."""
    decisions = state.get("decisions", [])
    actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]
    if actionable:
        return "options"
    return "report"


def build_graph() -> StateGraph:
    """Build and compile the pipeline graph.

    Flow:
        screener -> data_collector -> institutional -> technical -> pattern -> decision
                                                                                |
                                                           (BUY/SELL?) ---> options -> report
                                                           (all HOLD?) --------------> report
    """
    graph = StateGraph(PipelineState)

    graph.add_node("screener", run_screener)
    graph.add_node("data_collector", run_data_collector)
    graph.add_node("institutional", run_institutional)
    graph.add_node("technical", run_technical)
    graph.add_node("pattern", run_pattern)
    graph.add_node("decision", run_decision)
    graph.add_node("options", run_options)
    graph.add_node("report", run_report)

    graph.set_entry_point("screener")
    graph.add_edge("screener", "data_collector")
    graph.add_edge("data_collector", "institutional")
    graph.add_edge("institutional", "technical")
    graph.add_edge("technical", "pattern")
    graph.add_edge("pattern", "decision")

    graph.add_conditional_edges(
        "decision",
        _should_run_options,
        {"options": "options", "report": "report"},
    )
    graph.add_edge("options", "report")
    graph.add_edge("report", END)

    return graph.compile()
