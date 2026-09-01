"""LangGraph workflow definition and routing policy."""

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.analyst_agent import analyze_node
from app.agents.auditor_agent import audit_node
from app.agents.narrator_agent import narrator_node
from app.agents.retriever_agent import retrieve_node
from app.core.state import AgentState


def human_review_node(state: AgentState) -> dict[str, Any]:
    """Record the reviewer decision before routing approval or revision."""
    if state.get("human_approved"):
        return {"graph_history": [{"node": "human_review", "status": "approved"}]}
    history = state.get("graph_history", [])
    review_event = history[-1] if history else {}
    report = state.get("compliance_report", {})
    revised_report = {
        **report,
        "status": "REVISE",
        "audit_notes": [
            *report.get("audit_notes", []),
            review_event.get("comment", "Human reviewer requested revision."),
        ],
        "revision_target": report.get("revision_target") or "retrieve",
    }
    return {
        "compliance_report": revised_report,
        "revision_target": revised_report["revision_target"],
        "graph_history": [{"node": "human_review", "status": "revision_requested"}],
    }


def route_after_human_review(state: AgentState) -> Literal["retrieve", "analyze", "end"]:
    """Continue approved reports or route rejected reports back into the loop."""
    if state.get("human_approved"):
        return "end"
    return "retrieve" if state.get("revision_target") == "retrieve" else "analyze"


def route_after_audit(state: AgentState) -> Literal["retrieve", "analyze", "narrator"]:
    """Route revisions while enforcing the maximum iteration guardrail."""
    report = state.get("compliance_report", {})
    if report.get("status") == "PASS":
        return "narrator"
    if state.get("iteration_count", 0) >= 3:
        return "narrator"
    return "retrieve" if report.get("revision_target") == "retrieve" else "analyze"


def build_graph():
    """Build a compiled, memory-checkpointed research graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("narrator", narrator_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "audit")
    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {"retrieve": "retrieve", "analyze": "analyze", "narrator": "narrator"},
    )
    workflow.add_edge("narrator", "human_review")
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"retrieve": "retrieve", "analyze": "analyze", "end": END},
    )
    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
