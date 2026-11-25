"""Analyst Agent: 26-feature operand role scorers (p0, p1), propose(), and deterministic executor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from augus.parser import CandidateOperand, EvidenceParser
from augus.planner import EXECUTABLE_TEMPLATES, TEMPLATES, PlannerOutput
from augus.reranker import RerankedUnit
from augus.retriever import tokenize

_TEMPLATE_INDEX = {t: i for i, t in enumerate(TEMPLATES)}


def execute_program(template: str, a: float, b: float) -> tuple[float | None, str, str | None]:
    """Deterministically execute a symbolic program for one of the 5 executable templates.

    Returns:
        (result_or_none, canonical_program_str, error_or_none)
    """
    if template == "ratio":
        prog = f"divide({a}, {b})"
        if abs(b) < 1e-12:
            return None, prog, "division_by_zero"
        return a / b, prog, None

    if template == "percent change":
        prog = f"subtract({a}, {b}), divide(#0, {b})"
        if abs(b) < 1e-12:
            return None, prog, "division_by_zero"
        return (a - b) / b, prog, None

    if template == "difference":
        prog = f"subtract({a}, {b})"
        return a - b, prog, None

    if template == "sum":
        prog = f"add({a}, {b})"
        return a + b, prog, None

    if template == "product":
        prog = f"multiply({a}, {b})"
        return a * b, prog, None

    return None, "", "unsupported_template"


def operand_features(
    question: str,
    cand: CandidateOperand,
    planner_out: PlannerOutput,
    same_unit_count: int,
) -> np.ndarray:
    """Compute the 26-feature vector for a candidate operand."""
    q_tokens = set(tokenize(question))
    ctx_tokens = tokenize(cand.context)
    ctx_set = set(ctx_tokens)
    q_list = tokenize(question)
    q_bigrams = set(zip(q_list, q_list[1:]))
    ctx_bigrams = set(zip(ctx_tokens, ctx_tokens[1:]))

    row_tokens = set(tokenize(cand.row_label))
    col_tokens = set(tokenize(cand.col_header))

    ctx_uni = len(q_tokens & ctx_set) / max(len(q_tokens), 1)
    row_uni = len(q_tokens & row_tokens) / max(len(row_tokens), 1) if row_tokens else 0.0
    col_uni = len(q_tokens & col_tokens) / max(len(col_tokens), 1) if col_tokens else 0.0
    ctx_bi = len(q_bigrams & ctx_bigrams) / max(len(q_bigrams), 1)

    q_years = planner_out.question_years
    q_year_set = set(q_years)
    cand_years = {cand.col_year} if cand.col_year is not None else set(cand.unit_years)

    if q_year_set and cand_years:
        year_match = 1.0 if (q_year_set & cand_years) else 0.0
        year_mismatch = 1.0 if not (q_year_set & cand_years) else 0.0
    else:
        year_match = 0.0
        year_mismatch = 0.0

    is_first_q_year = 1.0 if (len(q_years) >= 1 and cand.col_year == q_years[0]) else 0.0
    is_second_q_year = 1.0 if (len(q_years) >= 2 and cand.col_year == q_years[1]) else 0.0

    is_small_int = 1.0 if (abs(cand.value) < 100.0 and float(cand.value).is_integer()) else 0.0
    has_decimal = 1.0 if "." in cand.raw_str else 0.0

    tpl_id = float(_TEMPLATE_INDEX.get(planner_out.template, len(TEMPLATES) - 1)) / max(len(TEMPLATES) - 1, 1)
    tpl_needs_order = 1.0 if planner_out.template in {"ratio", "percent change", "difference"} else 0.0

    feats = np.array(
        [
            cand.unit_score,
            1.0 / (1.0 + cand.unit_rank),
            cand.unit_bm25,
            cand.unit_tfidf,
            1.0 if cand.unit_type == "table" else 0.0,
            1.0 if cand.is_percent else 0.0,
            1.0 if cand.value < 0 else 0.0,
            math.log1p(abs(cand.value)),
            is_small_int,
            has_decimal,
            1.0 if cand.scale == "thousand" else 0.0,
            1.0 if cand.scale == "million" else 0.0,
            1.0 if cand.scale == "billion" else 0.0,
            ctx_uni,
            row_uni,
            col_uni,
            ctx_bi,
            year_match,
            year_mismatch,
            is_first_q_year,
            is_second_q_year,
            float(cand.col_index) / 10.0,
            float(cand.char_pos) / 500.0,
            float(min(same_unit_count, 20)) / 20.0,
            planner_out.pi_t,
            tpl_id * 0.5 + tpl_needs_order * 0.5,
        ],
        dtype=np.float64,
    )
    return feats


@dataclass
class Proposal:
    """Structured output of AnalystAgent.propose()."""

    template: str
    program: str
    exe_ans: float | None
    operand_0: CandidateOperand | None
    operand_1: CandidateOperand | None
    p0: float
    p1: float
    evidence_unit_ids: list[str] = field(default_factory=list)
    execution_error: str | None = None

    def to_operand_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for role, op in (("arg0", self.operand_0), ("arg1", self.operand_1)):
            if op is not None:
                sources.append(
                    {
                        "role": role,
                        "unit_id": op.unit_id,
                        "value": op.value,
                        "raw_str": op.raw_str,
                        "context": op.context,
                        "col_year": op.col_year,
                        "unit_years": sorted(op.unit_years),
                    }
                )
        return sources


class AnalystAgent:
    """26-feature GBT role scorers (p0, p1; 400 rounds, lr=0.06, L2=1.0) + symbolic executor."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.parser = EvidenceParser()
        self.scorer_0 = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        self.scorer_1 = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> "AnalystAgent":
        """Fit role scorers p0 and p1 on out-of-fold reranked candidate features."""
        self.scorer_0.fit(X, y0)
        self.scorer_1.fit(X, y1)
        self._fitted = True
        return self

    def extract_candidates(
        self,
        reranked_units: list[RerankedUnit],
    ) -> tuple[list[CandidateOperand], dict[str, int]]:
        """Extract candidate operands from reranked evidence units."""
        candidates: list[CandidateOperand] = []
        unit_counts: dict[str, int] = {}
        for ru in reranked_units:
            u = ru.retrieved.unit
            ops = self.parser.extract_operands(u)
            unit_counts[u.unit_id] = len(ops)
            for op in ops:
                op.unit_rank = ru.rerank_rank
                op.unit_score = ru.rerank_score
                op.unit_bm25 = ru.retrieved.bm25_norm
                op.unit_tfidf = ru.retrieved.tfidf_norm
                candidates.append(op)
        return candidates, unit_counts

    def propose(
        self,
        question: str,
        planner_out: PlannerOutput,
        reranked_units: list[RerankedUnit],
    ) -> Proposal:
        """Select (i*, j*) = argmax_{i != j} p0(c_i) * p1(c_j) and execute program."""
        unit_ids = [ru.retrieved.unit.unit_id for ru in reranked_units]
        if planner_out.template not in EXECUTABLE_TEMPLATES:
            return Proposal(
                template=planner_out.template,
                program="",
                exe_ans=None,
                operand_0=None,
                operand_1=None,
                p0=0.0,
                p1=0.0,
                evidence_unit_ids=unit_ids,
                execution_error="non_executable_template",
            )

        candidates, unit_counts = self.extract_candidates(reranked_units)
        if len(candidates) < 2:
            return Proposal(
                template=planner_out.template,
                program="",
                exe_ans=None,
                operand_0=None,
                operand_1=None,
                p0=0.0,
                p1=0.0,
                evidence_unit_ids=unit_ids,
                execution_error="insufficient_operands",
            )

        X = np.vstack(
            [
                operand_features(question, c, planner_out, unit_counts.get(c.unit_id, 1))
                for c in candidates
            ]
        )
        if self._fitted:
            p0_vec = self.scorer_0.predict_proba(X)[:, 1]
            p1_vec = self.scorer_1.predict_proba(X)[:, 1]
        else:
            p0_vec = np.array([max(c.unit_score, 1e-4) for c in candidates], dtype=np.float64)
            p1_vec = np.array([max(c.unit_score, 1e-4) for c in candidates], dtype=np.float64)

        # Select (i*, j*) = argmax_{i != j} p0(c_i) * p1(c_j)
        pair_scores = np.outer(p0_vec, p1_vec)
        np.fill_diagonal(pair_scores, -1.0)
        flat_idx = int(np.argmax(pair_scores))
        i_star, j_star = divmod(flat_idx, len(candidates))

        op0 = candidates[i_star]
        op1 = candidates[j_star]
        p0_val = float(p0_vec[i_star])
        p1_val = float(p1_vec[j_star])

        exe_ans, prog, err = execute_program(planner_out.template, op0.value, op1.value)
        cited_ids = list(dict.fromkeys([op0.unit_id, op1.unit_id]))
        return Proposal(
            template=planner_out.template,
            program=prog,
            exe_ans=exe_ans,
            operand_0=op0,
            operand_1=op1,
            p0=p0_val,
            p1=p1_val,
            evidence_unit_ids=cited_ids,
            execution_error=err,
        )
