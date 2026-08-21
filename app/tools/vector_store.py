"""Small hybrid retrieval abstraction with optional Qdrant integration."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class HybridVectorStore:
    """In-memory dense-like token search with BM25-style lexical scoring.

    The interface is intentionally backend-neutral so Qdrant can be introduced
    behind the same contract without changing agent orchestration.
    """

    def __init__(self, embeddings: Any | None = None) -> None:
        self._documents: list[dict[str, Any]] = []
        self._embeddings = embeddings
        self._vectors: list[list[float]] = []

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self._documents.extend(documents)
        if self._embeddings:
            self._vectors.extend(self._embeddings.embed_documents([document.get("text", "") for document in documents]))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def hybrid_search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Rank documents with local dense similarity plus lexical overlap."""
        query_tokens = self._tokens(query)
        query_counts = Counter(query_tokens)
        query_vector = self._embeddings.embed_query(query) if self._embeddings else None
        scored = []
        for index, document in enumerate(self._documents):
            tokens = self._tokens(document.get("text", ""))
            counts = Counter(tokens)
            lexical = sum(min(counts[token], frequency) for token, frequency in query_counts.items())
            phrase = 0.25 if query.lower() in document.get("text", "").lower() else 0.0
            dense = 0.0
            if query_vector and index < len(self._vectors):
                dense = sum(left * right for left, right in zip(query_vector, self._vectors[index]))
            scored.append((lexical + phrase + dense, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in scored[:limit] if score > 0] or [
            document for _, document in scored[:limit]
        ]
