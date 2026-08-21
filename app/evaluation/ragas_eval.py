"""RAGAS-compatible evaluation runner with a dependency-free fallback."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agents.analyst_agent import analyze_documents
from app.tools.sec_tools import fetch_and_chunk_filing


async def evaluate_case(query: str, ticker: str = "DEMO") -> dict[str, float]:
    """Return context precision, faithfulness, and answer relevance proxies.

    Install ``ragas`` and adapt this dataset to use model-based scoring in CI;
    these deterministic checks keep local runs reproducible and offline.
    """
    documents = await fetch_and_chunk_filing(ticker)
    analysis = analyze_documents(documents)
    relevant = sum(word.lower() in " ".join(doc["text"] for doc in documents).lower() for word in query.split())
    precision = relevant / max(len(query.split()), 1)
    faithfulness = float(bool(analysis["evidence_chunk_ids"]))
    relevance = float(bool(analysis["summary"]))
    return {"context_precision": round(min(precision, 1.0), 3), "faithfulness": faithfulness, "answer_relevance": relevance}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate_case("revenue growth and risk factors")), indent=2))
