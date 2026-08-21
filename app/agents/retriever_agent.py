"""Retriever agent: query decomposition, SEC acquisition, and hybrid ranking."""

from typing import Any

from langchain_core.messages import AIMessage

from app.tools.sec_tools import fetch_and_chunk_filing
from app.tools.vector_store import HybridVectorStore
from app.core.providers import get_embeddings


def decompose_query(query: str) -> list[str]:
    """Produce focused retrieval questions from a compound financial query."""
    parts = [part.strip() for part in query.replace("?", ".").split(".") if part.strip()]
    return parts or [query]


async def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve relevant filing chunks and record the retrieval iteration."""
    ticker = state.get("company_ticker", "DEMO")
    documents = await fetch_and_chunk_filing(ticker)
    try:
        store = HybridVectorStore(embeddings=get_embeddings())
    except (OSError, RuntimeError, ImportError):
        store = HybridVectorStore()
    store.add_documents(documents)
    queries = decompose_query(state["query"])
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for document in store.hybrid_search(query):
            chunk_id = document["metadata"]["chunk_id"]
            if chunk_id not in seen:
                ranked.append(document)
                seen.add(chunk_id)
    iteration = state.get("iteration_count", 0) + 1
    return {
        "retrieved_docs": ranked,
        "iteration_count": iteration,
        "messages": [AIMessage(content=f"Retrieved {len(ranked)} SEC filing chunks.")],
        "graph_history": [{"node": "retrieve", "iteration": iteration}],
    }
