"""Tests for deterministic finance and retrieval tools."""

import pytest

from app.tools.finance_calc import calculate_metrics
from app.agents.analyst_agent import analyze_documents
from app.tools.sec_tools import fetch_and_chunk_filing
from app.tools.vector_store import HybridVectorStore


def test_financial_metrics_are_deterministic() -> None:
    metrics = calculate_metrics({"revenue": 120.0, "prior_revenue": 100.0, "operating_income": 24.0, "total_debt": 30.0, "total_assets": 100.0})
    assert metrics == {"revenue_growth": 0.2, "operating_margin": 0.2, "debt_to_assets": 0.3}


def test_hybrid_search_preserves_provenance() -> None:
    store = HybridVectorStore()
    document = {"text": "debt covenant disclosure", "metadata": {"section": "Item 1A"}}
    store.add_documents([document])
    assert store.hybrid_search("debt covenant")[0]["metadata"]["section"] == "Item 1A"


@pytest.mark.asyncio
async def test_demo_sec_chunks_have_metadata() -> None:
    documents = await fetch_and_chunk_filing("DEMO")
    assert documents
    assert {"source", "cik", "filing_date", "section", "chunk_id"} <= documents[0]["metadata"].keys()


@pytest.mark.asyncio
async def test_demo_analysis_extracts_comparative_revenue() -> None:
    documents = await fetch_and_chunk_filing("DEMO")
    analysis = analyze_documents(documents)
    assert analysis["metrics"]["revenue_growth"] == 0.25
