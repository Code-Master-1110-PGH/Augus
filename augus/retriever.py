"""Retriever Agent: BM25 (k1=1.5, b=0.75) + TF-IDF cosine similarity with per-filing min-max fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from augus.parser import EvidenceUnit

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer preserving percentages and decimals."""
    return _TOKEN_RE.findall(text.lower())


def min_max_normalize(scores: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalize scores into [0.0, 1.0] per filing."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr, dtype=np.float64)


@dataclass
class RetrievedUnit:
    """Evidence unit paired with normalized retrieval scores."""

    unit: EvidenceUnit
    bm25_raw: float
    tfidf_raw: float
    bm25_norm: float
    tfidf_norm: float
    hybrid_score: float
    bm25_rank: int
    tfidf_rank: int


class RetrieverAgent:
    """Per-filing BM25 + TF-IDF hybrid retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, alpha: float = 0.5) -> None:
        self.k1 = k1
        self.b = b
        self.alpha = alpha

    def _bm25_scores(self, query_tokens: list[str], doc_tokens_list: list[list[str]]) -> np.ndarray:
        n_docs = len(doc_tokens_list)
        if n_docs == 0:
            return np.array([], dtype=np.float64)

        doc_lens = np.array([len(toks) for toks in doc_tokens_list], dtype=np.float64)
        avgdl = float(doc_lens.mean()) if doc_lens.sum() > 0 else 1.0

        df: Counter[str] = Counter()
        tfs: list[Counter[str]] = []
        for toks in doc_tokens_list:
            c = Counter(toks)
            tfs.append(c)
            for tok in c.keys():
                df[tok] += 1

        scores = np.zeros(n_docs, dtype=np.float64)
        q_counts = Counter(query_tokens)
        for q_tok in q_counts.keys():
            n_q = df.get(q_tok, 0)
            if n_q == 0:
                continue
            idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for idx, tf_map in enumerate(tfs):
                tf = tf_map.get(q_tok, 0)
                if tf == 0:
                    continue
                dl = doc_lens[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

    def _tfidf_scores(self, query: str, doc_texts: list[str]) -> np.ndarray:
        if not doc_texts:
            return np.array([], dtype=np.float64)
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b")
        try:
            mat = vec.fit_transform(doc_texts + [query])
        except ValueError:
            return np.zeros(len(doc_texts), dtype=np.float64)
        sims = cosine_similarity(mat[-1], mat[:-1])[0]
        return np.asarray(sims, dtype=np.float64)

    def score_units(
        self,
        query: str,
        units: list[EvidenceUnit],
        mode: Literal["hybrid", "bm25", "tfidf"] = "hybrid",
    ) -> list[RetrievedUnit]:
        """Score and sort all evidence units of a filing."""
        if not units:
            return []

        q_tokens = tokenize(query)
        doc_tokens_list = [tokenize(u.text) for u in units]
        doc_texts = [u.text for u in units]

        bm25_raw = self._bm25_scores(q_tokens, doc_tokens_list)
        tfidf_raw = self._tfidf_scores(query, doc_texts)

        bm25_norm = min_max_normalize(bm25_raw)
        tfidf_norm = min_max_normalize(tfidf_raw)

        bm25_order = np.argsort(-bm25_norm, kind="mergesort")
        tfidf_order = np.argsort(-tfidf_norm, kind="mergesort")
        bm25_ranks = np.empty(len(units), dtype=int)
        tfidf_ranks = np.empty(len(units), dtype=int)
        bm25_ranks[bm25_order] = np.arange(len(units))
        tfidf_ranks[tfidf_order] = np.arange(len(units))

        if mode == "bm25":
            combined = bm25_norm
        elif mode == "tfidf":
            combined = tfidf_norm
        else:
            combined = self.alpha * bm25_norm + (1.0 - self.alpha) * tfidf_norm

        results = [
            RetrievedUnit(
                unit=u,
                bm25_raw=float(bm25_raw[i]),
                tfidf_raw=float(tfidf_raw[i]),
                bm25_norm=float(bm25_norm[i]),
                tfidf_norm=float(tfidf_norm[i]),
                hybrid_score=float(combined[i]),
                bm25_rank=int(bm25_ranks[i]),
                tfidf_rank=int(tfidf_ranks[i]),
            )
            for i, u in enumerate(units)
        ]
        results.sort(key=lambda r: (-r.hybrid_score, -r.bm25_norm, r.unit.index))
        return results
