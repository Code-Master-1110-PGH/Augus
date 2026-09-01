"""Narrator agent for generating descriptive prose explanations of research findings."""

from typing import Any

from langchain_core.messages import AIMessage


def generate_narrative(state: dict[str, Any]) -> str:
    """
    Generate a comprehensive, multi-paragraph narrative explanation answering the research query.
    
    Combines financial metrics, risk analysis, and evidence into conversational prose that
    directly addresses the user's original research question.
    
    Args:
        state: Current agent state containing query, analysis, and compliance report
        
    Returns:
        Formatted multi-paragraph narrative explanation
    """
    query = state.get("query", "").strip()
    analysis = state.get("financial_analysis", {})
    documents = state.get("retrieved_docs", [])
    report = state.get("compliance_report", {})
    
    metrics = analysis.get("metrics", {})
    risks = analysis.get("risk_factors", [])
    sentiment = analysis.get("sentiment", "pending")
    
    # Extract key metrics for narrative
    revenue_growth = metrics.get("revenue_growth")
    operating_margin = metrics.get("operating_margin")
    debt_to_assets = metrics.get("debt_to_assets")
    
    # Build opening paragraph addressing the research query
    opening = f"Based on a comprehensive analysis of {len(documents)} evidence chunks from SEC filings, "
    
    if query.lower().startswith("assess"):
        opening += f"this research addresses your assessment request regarding {state.get('company_ticker', 'the company')}'s financial position and risk profile."
    elif "risk" in query.lower():
        opening += f"this research identifies and evaluates the key risk factors disclosed by {state.get('company_ticker', 'the company')}."
    elif "performance" in query.lower():
        opening += f"this research examines the operational and financial performance metrics for {state.get('company_ticker', 'the company')}."
    else:
        opening += f"here are the key findings relevant to your question: {query}"
    
    # Build financial metrics paragraph
    metrics_paragraph = "Financial Performance Overview: "
    if revenue_growth is not None:
        metrics_paragraph += f"The company demonstrated a revenue growth rate of {revenue_growth*100:.1f}% between reporting periods, "
    if operating_margin is not None:
        metrics_paragraph += f"with an operating margin of {operating_margin*100:.1f}% reflecting operational efficiency. "
    if debt_to_assets is not None:
        metrics_paragraph += f"The debt-to-assets ratio stands at {debt_to_assets*100:.1f}%, indicating {'moderate' if debt_to_assets < 0.5 else 'significant' if debt_to_assets < 0.75 else 'elevated'} financial leverage. "
    
    if not (revenue_growth or operating_margin or debt_to_assets):
        metrics_paragraph += "The analyst is still processing the retrieved evidence for quantitative metrics. "
    
    # Build risk assessment paragraph
    risk_paragraph = "Risk Assessment & Disclosures: "
    if risks:
        risk_list = ", ".join(risks)
        risk_paragraph += f"The company disclosed exposure to the following key risk factors: {risk_list}. "
        risk_paragraph += f"These disclosures suggest a {sentiment} outlook requiring close monitoring. "
    else:
        risk_paragraph += "No major risk factors were explicitly flagged in the retrieved filings. "
    
    # Build sentiment and recommendation paragraph
    sentiment_paragraph = f"Analyst Sentiment & Recommendation: "
    if sentiment == "cautious":
        sentiment_paragraph += "The overall assessment is cautious, reflecting material risk exposures or operational headwinds that warrant careful due diligence. "
    elif sentiment == "neutral":
        sentiment_paragraph += "The overall assessment is neutral, with balanced growth and risk factors. "
    else:
        sentiment_paragraph += "The assessment remains pending further analysis of the retrieved evidence. "
    
    # Build compliance paragraph
    compliance_paragraph = "Verification Status: "
    audit_status = report.get("status", "PENDING")
    audit_notes = report.get("audit_notes", [])
    
    if audit_status == "PASS":
        compliance_paragraph += "This analysis has passed all compliance checks for disclosure completeness, evidence provenance, and risk factor reporting. "
    elif audit_status == "REVISE":
        issues = "; ".join(audit_notes) if audit_notes else "some completeness or provenance concerns"
        compliance_paragraph += f"The analysis flagged {issues} and may require revision to strengthen the evidence base. "
    else:
        compliance_paragraph += "The compliance audit is ongoing to verify evidence quality and completeness. "
    
    # Combine paragraphs into full narrative
    narrative = "\n\n".join([
        opening,
        metrics_paragraph,
        risk_paragraph,
        sentiment_paragraph,
        compliance_paragraph,
    ])
    
    return narrative.strip()


async def narrator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate descriptive narrative and append to state."""
    narrative = generate_narrative(state)
    return {
        "descriptive_answer": narrative,
        "messages": [AIMessage(content="Generated comprehensive narrative explanation of research findings.")],
        "graph_history": [{"node": "narrator", "status": "COMPLETE"}],
    }
