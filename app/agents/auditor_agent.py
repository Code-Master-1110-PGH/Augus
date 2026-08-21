"""Compliance auditor with targeted revision routing."""

from typing import Any

from langchain_core.messages import AIMessage


def audit_findings(state: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic completeness and grounding rules to analyst output."""
    analysis = state.get("financial_analysis", {})
    documents = state.get("retrieved_docs", [])
    notes: list[str] = []
    target = "analyze"
    if not documents:
        notes.append("No filing evidence was retrieved.")
        target = "retrieve"
    if not analysis.get("metrics"):
        notes.append("No deterministic financial metrics were available.")
        target = "analyze" if documents else "retrieve"
    if not analysis.get("evidence_chunk_ids"):
        notes.append("Analysis lacks chunk-level provenance.")
        target = "analyze"
    risks = analysis.get("risk_factors", [])
    if not risks:
        notes.append("Risk factor disclosure could not be confirmed.")
        target = "retrieve"
    status = "REVISE" if notes else "PASS"
    return {
        "status": status,
        "audit_notes": notes or ["Required disclosure and provenance checks passed."],
        "flagged_risks": risks,
        "revision_target": target if status == "REVISE" else None,
    }


async def audit_node(state: dict[str, Any]) -> dict[str, Any]:
    """Audit findings and emit a machine-readable compliance report."""
    report = audit_findings(state)
    return {
        "compliance_report": report,
        "revision_target": report.get("revision_target"),
        "messages": [AIMessage(content=f"Compliance audit status: {report['status']}.")],
        "graph_history": [{"node": "audit", "status": report["status"]}],
    }
