"""Reranker Agent: 15-feature Gradient Boosted Trees returning top-5 evidence units."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from augus.parser import EvidenceParser, extract_years
from augus.retriever import RetrievedUnit, tokenize

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "at",
        "by", "from", "with", "as", "is", "was", "were", "are", "be", "what",
        "which", "how", "much", "many", "did", "does", "do", "between",
    }
)


def _content_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def reranker_features(
    question: str,
    retrieved: RetrievedUnit,
    parser: EvidenceParser | None = None,
) -> np.ndarray:
    """Compute the 15-feature vector for a (question, retrieved_unit) pair."""
    q_tokens = tokenize(question)
    q_content = set(_content_tokens(q_tokens))
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    q_trigrams = {" ".join(q_tokens[i : i + 3]) for i in range(max(0, len(q_tokens) - 2))}
    q_years = set(extract_years(question))

    u = retrieved.unit
    u_tokens = tokenize(u.text)
    u_set = set(u_tokens)
    u_bigrams = set(zip(u_tokens, u_tokens[1:]))
    u_joined = " ".join(u_tokens)

    unigram_overlap = len(q_content & u_set) / max(len(q_content), 1)
    bigram_overlap = len(q_bigrams & u_bigrams) / max(len(q_bigrams), 1)
    exact_trigram = 1.0 if any(tg in u_joined for tg in q_trigrams) else 0.0

    if q_years and u.years:
        year_overlap = len(q_years & u.years) / len(q_years)
        year_mismatch = 1.0 if len(q_years & u.years) == 0 else 0.0
    else:
        year_overlap = 0.0
        year_mismatch = 0.0

    if parser is not None:
        cached = getattr(u, "_cached_num_operands", None)
        if cached is None:
            cached = float(min(len(parser.extract_operands(u)), 20)) / 20.0
            setattr(u, "_cached_num_operands", cached)
        num_operands = cached
    else:
        num_operands = float(min(sum(1 for t in u_tokens if any(ch.isdigit() for ch in t)), 20)) / 20.0

    u_lower = u.text.lower()
    has_percent = 1.0 if ("%" in u_lower or "percent" in u_lower) else 0.0
    has_currency = 1.0 if ("$" in u_lower or "million" in u_lower or "billion" in u_lower) else 0.0
    unit_len_log = math.log1p(len(u_tokens))

    feats = np.array(
        [
            retrieved.bm25_norm,
            retrieved.tfidf_norm,
            retrieved.hybrid_score,
            1.0 / (1.0 + retrieved.bm25_rank),
            1.0 / (1.0 + retrieved.tfidf_rank),
            1.0 if u.unit_type == "table" else 0.0,
            unigram_overlap,
            bigram_overlap,
            exact_trigram,
            year_overlap,
            year_mismatch,
            num_operands,
            has_percent,
            has_currency,
            unit_len_log,
        ],
        dtype=np.float64,
    )
    return feats


@dataclass
class RerankedUnit:
    """Evidence unit with reranker probability and rank."""

    retrieved: RetrievedUnit
    rerank_score: float
    rerank_rank: int


class RerankerAgent:
    """15-feature GBT reranker (300 rounds, max 31 leaves, lr=0.08) returning top-5 units."""

    def __init__(self, seed: int = 42, top_k: int = 5) -> None:
        self.seed = seed
        self.top_k = top_k
        self.parser = EvidenceParser()
        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            max_leaf_nodes=31,
            learning_rate=0.08,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RerankerAgent":
        """Fit the GBT reranker on (features, binary_gold_label)."""
        self.model.fit(X, y)
        self._fitted = True
        return self

    def rerank(
        self,
        question: str,
        retrieved_units: list[RetrievedUnit],
        top_k: int | None = None,
    ) -> list[RerankedUnit]:
        """Score retrieved units and return top-k reranked evidence units."""
        if not retrieved_units:
            return []
        k = top_k if top_k is not None else self.top_k
        if self._fitted:
            X = np.vstack([reranker_features(question, r, self.parser) for r in retrieved_units])
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = np.array([r.hybrid_score for r in retrieved_units], dtype=np.float64)

        order = np.argsort(-probs, kind="mergesort")[:k]
        return [
            RerankedUnit(
                retrieved=retrieved_units[idx],
                rerank_score=float(probs[idx]),
                rerank_rank=rank,
            )
            for rank, idx in enumerate(order)
        ]
