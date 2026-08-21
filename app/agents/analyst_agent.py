"""Analyst agent for deterministic metrics and grounded qualitative findings."""

import re
from typing import Any

from langchain_core.messages import AIMessage

from app.tools.finance_calc import calculate_metrics


def _first_amount(text: str, label: str) -> float | None:
    pattern = rf"{re.escape(label)}[^\n$]{{0,160}}\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def analyze_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract known demo filing amounts and summarize risk disclosures."""
    text = " ".join(document.get("text", "") for document in documents)
    values = {
        key: amount
        for key, label in {
            "revenue": "revenue",
            "prior_revenue": "for 2023",
            "operating_income": "operating income",
            "total_debt": "total debt",
            "total_assets": "total assets",
        }.items()
        if (amount := _first_amount(text, label)) is not None
    }
    revenue_pair = re.search(
        r"revenue[^.\n]*?\$?([0-9][0-9,]*)\s*million[^.\n]*?2024[^.\n]*?\$?([0-9][0-9,]*)\s*million[^.\n]*?2023",
        text,
        re.IGNORECASE,
    )
    if revenue_pair:
        values["revenue"] = float(revenue_pair.group(1).replace(",", ""))
        values["prior_revenue"] = float(revenue_pair.group(2).replace(",", ""))
    risks = [
        risk
        for risk in ("liquidity", "interest-rate", "cybersecurity", "customer concentration")
        if risk in text.lower()
    ]
    metrics = calculate_metrics(values)
    return {
        "metrics": metrics,
        "sentiment": "cautious" if risks else "neutral",
        "risk_factors": risks,
        "summary": "Analysis is grounded in retrieved SEC filing chunks.",
        "evidence_chunk_ids": [doc["metadata"]["chunk_id"] for doc in documents],
    }


async def analyze_node(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze retrieved evidence and append an auditable message."""
    analysis = analyze_documents(state.get("retrieved_docs", []))
    return {
        "financial_analysis": analysis,
        "messages": [AIMessage(content="Computed deterministic financial metrics and risk sentiment.")],
        "graph_history": [{"node": "analyze", "iteration": state.get("iteration_count", 0)}],
    }
