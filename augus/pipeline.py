"""End-to-end 5-agent AUGUS pipeline orchestrator and OOF trainer."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from augus.analyst import AnalystAgent, Proposal, operand_features
from augus.compliance import ComplianceAgent, ComplianceResult
from augus.parser import EvidenceParser, EvidenceUnit
from augus.planner import EXECUTABLE_TEMPLATES, PlannerAgent, PlannerOutput, classify_program_template
from augus.reranker import RerankedUnit, RerankerAgent, reranker_features
from augus.retriever import RetrieverAgent

_STEP_RE = re.compile(r"([a-zA-Z_]+)\(([^()]*)\)")


def parse_gold_operands(program: str) -> tuple[float | None, float | None]:
    """Extract the two numeric gold operands (arg0, arg1) from an executable FinQA program."""
    steps = [(m.group(1).lower(), [a.strip() for a in m.group(2).split(",")]) for m in _STEP_RE.finditer(program or "")]
    if not steps:
        return None, None
    args = steps[0][1]
    if len(args) != 2:
        return None, None

    vals: list[float] = []
    for a in args:
        if a.startswith("#") or a.startswith("const_"):
            return None, None
        s = a.strip()
        is_pct = s.endswith("%")
        if is_pct:
            s = s[:-1].strip()
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            return None, None
        if is_pct:
            v = v / 100.0
        vals.append(v)
    return vals[0], vals[1]


def _match_gold_value(cand_val: float, gold_val: float) -> bool:
    if abs(cand_val - gold_val) <= 1e-4 * max(1.0, abs(gold_val)):
        return True
    # Handle percent-scale variants (e.g. 15% parsed as 0.15 vs 15.0)
    if abs(cand_val * 100.0 - gold_val) <= 1e-3 * max(1.0, abs(gold_val)):
        return True
    if abs(cand_val / 100.0 - gold_val) <= 1e-4 * max(1.0, abs(gold_val)):
        return True
    return False


class AUGUSPipeline:
    """Coordinates Parser, Planner, Retriever, Reranker, Analyst, and Compliance Agents."""

    def __init__(self, seed: int = 42, top_k: int = 5, n_folds: int = 5) -> None:
        self.seed = seed
        self.top_k = top_k
        self.n_folds = n_folds
        self.parser = EvidenceParser()
        self.planner = PlannerAgent(seed=seed)
        self.retriever = RetrieverAgent(k1=1.5, b=0.75, alpha=0.5)
        self.reranker = RerankerAgent(seed=seed, top_k=top_k)
        self.analyst = AnalystAgent(seed=seed)
        self.compliance = ComplianceAgent(tau=0.0)

    def fit(self, train_records: list[dict[str, Any]]) -> "AUGUSPipeline":
        """Train Planner, Reranker, and Analyst (on OOF reranked candidates)."""
        questions = [r["qa"]["question"] for r in train_records]
        templates = [classify_program_template(r["qa"].get("program", "")) for r in train_records]
        self.planner.fit(questions, templates)

        # Parse & retrieve all training records once
        parsed_units: list[list[EvidenceUnit]] = []
        retrieved_all: list[list[Any]] = []
        rerank_X_per_rec: list[np.ndarray] = []
        rerank_y_per_rec: list[np.ndarray] = []

        for r in train_records:
            units = self.parser.parse_filing(r)
            parsed_units.append(units)
            ret = self.retriever.score_units(r["qa"]["question"], units, mode="hybrid")
            retrieved_all.append(ret)
            gold_inds = set((r["qa"].get("gold_inds") or {}).keys())
            if ret:
                X_r = np.vstack([reranker_features(r["qa"]["question"], ru, self.parser) for ru in ret])
                y_r = np.array([1 if ru.unit.unit_id in gold_inds else 0 for ru in ret], dtype=np.int32)
            else:
                X_r = np.empty((0, 15), dtype=np.float64)
                y_r = np.empty((0,), dtype=np.int32)
            rerank_X_per_rec.append(X_r)
            rerank_y_per_rec.append(y_r)

        # Fit full reranker
        valid_X = [x for x in rerank_X_per_rec if len(x) > 0]
        valid_y = [y for y in rerank_y_per_rec if len(y) > 0]
        self.reranker.fit(np.vstack(valid_X), np.concatenate(valid_y))

        # Out-of-fold reranked candidates for Analyst training
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        indices = np.arange(len(train_records))
        oof_reranked: list[list[RerankedUnit]] = [[] for _ in train_records]
        oof_planner: list[PlannerOutput | None] = [None for _ in train_records]

        for train_idx, val_idx in kf.split(indices):
            fold_reranker = RerankerAgent(seed=self.seed, top_k=self.top_k)
            X_tr = np.vstack([rerank_X_per_rec[i] for i in train_idx if len(rerank_X_per_rec[i]) > 0])
            y_tr = np.concatenate([rerank_y_per_rec[i] for i in train_idx if len(rerank_y_per_rec[i]) > 0])
            fold_reranker.fit(X_tr, y_tr)

            fold_planner = PlannerAgent(seed=self.seed)
            fold_planner.fit([questions[i] for i in train_idx], [templates[i] for i in train_idx])

            for i in val_idx:
                oof_reranked[i] = fold_reranker.rerank(questions[i], retrieved_all[i], top_k=self.top_k)
                oof_planner[i] = fold_planner.predict(questions[i])

        # Build Analyst training set from OOF reranked units
        analyst_X_list: list[np.ndarray] = []
        analyst_y0_list: list[int] = []
        analyst_y1_list: list[int] = []

        for i, r in enumerate(train_records):
            tpl = templates[i]
            if tpl not in EXECUTABLE_TEMPLATES:
                continue
            g0, g1 = parse_gold_operands(r["qa"].get("program", ""))
            if g0 is None or g1 is None:
                continue
            gold_inds = set((r["qa"].get("gold_inds") or {}).keys())
            ru_list = oof_reranked[i]
            p_out = oof_planner[i]
            if not ru_list or p_out is None:
                continue
            cands, unit_counts = self.analyst.extract_candidates(ru_list)
            for c in cands:
                in_gold = c.unit_id in gold_inds if gold_inds else True
                y0 = 1 if (in_gold and _match_gold_value(c.value, g0)) else 0
                y1 = 1 if (in_gold and _match_gold_value(c.value, g1)) else 0
                analyst_X_list.append(
                    operand_features(questions[i], c, p_out, unit_counts.get(c.unit_id, 1))
                )
                analyst_y0_list.append(y0)
                analyst_y1_list.append(y1)

        if analyst_X_list:
            self.analyst.fit(
                np.vstack(analyst_X_list),
                np.array(analyst_y0_list, dtype=np.int32),
                np.array(analyst_y1_list, dtype=np.int32),
            )

        return self

    def run_record(
        self,
        record: dict[str, Any],
        use_reranker: bool = True,
        retrieval_mode: str = "hybrid",
        use_planner: bool = True,
        oracle_gold_only: bool = False,
        tau: float | None = None,
        precomputed_units: list[EvidenceUnit] | None = None,
        precomputed_retrieved: list[Any] | None = None,
        precomputed_reranked: list[RerankedUnit] | None = None,
    ) -> tuple[PlannerOutput, list[RerankedUnit], Proposal, ComplianceResult]:
        """Run the full 5-agent pipeline on one FinQA record."""
        q = record["qa"]["question"]
        units = precomputed_units if precomputed_units is not None else self.parser.parse_filing(record)

        if use_planner:
            planner_out = self.planner.predict(q)
        else:
            # Uniform/heuristic fallback without trained planner (ablation)
            planner_out = PlannerOutput(
                template="ratio",
                pi_t=1.0 / 6.0,
                other_prob=1.0 / 6.0,
                probs={t: 1.0 / 6.0 for t in ("ratio", "percent change", "difference", "sum", "product", "other")},
                question_years=[],
            )

        if precomputed_reranked is not None:
            reranked = precomputed_reranked
        elif oracle_gold_only:
            gold_ids = set((record["qa"].get("gold_inds") or {}).keys())
            gold_units = [u for u in units if u.unit_id in gold_ids] or units[: self.top_k]
            retrieved = self.retriever.score_units(q, gold_units, mode="hybrid")
            reranked = [RerankedUnit(retrieved=ru, rerank_score=1.0, rerank_rank=idx) for idx, ru in enumerate(retrieved[: self.top_k])]
        else:
            retrieved = (
                precomputed_retrieved
                if precomputed_retrieved is not None
                else self.retriever.score_units(q, units, mode=retrieval_mode)  # type: ignore[arg-type]
            )
            if use_reranker:
                reranked = self.reranker.rerank(q, retrieved, top_k=self.top_k)
            else:
                reranked = [
                    RerankedUnit(retrieved=ru, rerank_score=ru.hybrid_score, rerank_rank=idx)
                    for idx, ru in enumerate(retrieved[: self.top_k])
                ]

        proposal = self.analyst.propose(q, planner_out, reranked)
        cited_units = [ru.retrieved.unit for ru in reranked]
        comp = self.compliance.verify(
            question_id=str(record.get("id", "")),
            planner_out=planner_out,
            proposal=proposal,
            cited_units=cited_units,
            tau=tau,
        )
        return planner_out, reranked, proposal, comp
