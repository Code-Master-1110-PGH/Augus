"""Compliance Agent: joint confidence kappa, 3 release checks, and SHA-256 JSONL audit log."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augus.analyst import Proposal
from augus.parser import CandidateOperand, EvidenceParser, EvidenceUnit
from augus.planner import EXECUTABLE_TEMPLATES, PlannerOutput


def compute_evidence_sha256(units: list[EvidenceUnit]) -> str:
    """Compute deterministic SHA-256 hex digest of ordered evidence unit texts."""
    canonical = "\n".join(f"{u.unit_id}:{u.text}" for u in units)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _operand_in_units(
    operand: CandidateOperand,
    unit_map: dict[str, EvidenceUnit],
    parser: EvidenceParser,
    tol: float = 1e-5,
) -> bool:
    unit = unit_map.get(operand.unit_id)
    if unit is None:
        return False
    parsed_ops = parser.extract_operands(unit)
    for po in parsed_ops:
        if abs(po.value - operand.value) <= tol * max(1.0, abs(po.value)):
            return True
    return False


def _temporal_coherent(
    planner_out: PlannerOutput,
    op0: CandidateOperand,
    op1: CandidateOperand,
) -> bool:
    q_years = set(planner_out.question_years)
    if not q_years:
        return True

    for op in (op0, op1):
        if op.col_year is not None and op.col_year not in q_years:
            return False
        if op.col_year is None and op.unit_years and not (op.unit_years & q_years):
            return False

    if len(q_years) >= 2 and op0.col_year is not None and op1.col_year is not None:
        if op0.col_year == op1.col_year and planner_out.template in {"percent change", "difference"}:
            return False

    return True


@dataclass
class ComplianceResult:
    """Result of the 3 compliance release checks and audit record."""

    question_id: str
    released: bool
    reason_code: str
    kappa: float
    evidence_unit_ids: list[str]
    evidence_sha256: str
    program: str
    exe_ans: float | None
    operand_sources: list[dict[str, Any]]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "evidence_unit_ids": self.evidence_unit_ids,
            "evidence_sha256": self.evidence_sha256,
            "program": self.program,
            "operand_sources": self.operand_sources,
            "kappa": round(self.kappa, 6),
            "released": self.released,
            "reason_code": self.reason_code,
        }


class ComplianceAgent:
    """Enforces operand grounding, temporal coherence, and kappa >= tau with JSONL logging."""

    def __init__(self, tau: float = 0.0) -> None:
        self.tau = tau
        self.parser = EvidenceParser()

    def verify(
        self,
        question_id: str,
        planner_out: PlannerOutput,
        proposal: Proposal,
        cited_units: list[EvidenceUnit],
        tau: float | None = None,
    ) -> ComplianceResult:
        """Compute kappa and enforce the 3 release checks."""
        threshold = self.tau if tau is None else tau
        kappa = float(planner_out.pi_t * proposal.p0 * proposal.p1)
        unit_map = {u.unit_id: u for u in cited_units}
        ordered_cited = [unit_map[uid] for uid in proposal.evidence_unit_ids if uid in unit_map]
        if not ordered_cited:
            ordered_cited = list(cited_units)
        sha256_hex = compute_evidence_sha256(ordered_cited)

        if proposal.template not in EXECUTABLE_TEMPLATES:
            return ComplianceResult(
                question_id=question_id,
                released=False,
                reason_code="ABSTAIN_OTHER_TEMPLATE",
                kappa=kappa,
                evidence_unit_ids=[u.unit_id for u in ordered_cited],
                evidence_sha256=sha256_hex,
                program=proposal.program,
                exe_ans=None,
                operand_sources=proposal.to_operand_sources(),
            )

        if proposal.exe_ans is None or proposal.operand_0 is None or proposal.operand_1 is None:
            return ComplianceResult(
                question_id=question_id,
                released=False,
                reason_code="ABSTAIN_EXECUTION_ERROR",
                kappa=kappa,
                evidence_unit_ids=[u.unit_id for u in ordered_cited],
                evidence_sha256=sha256_hex,
                program=proposal.program,
                exe_ans=None,
                operand_sources=proposal.to_operand_sources(),
            )

        # Check 1: Operand grounding
        if not (
            _operand_in_units(proposal.operand_0, unit_map, self.parser)
            and _operand_in_units(proposal.operand_1, unit_map, self.parser)
        ):
            return ComplianceResult(
                question_id=question_id,
                released=False,
                reason_code="ABSTAIN_UNGROUNDED_OPERAND",
                kappa=kappa,
                evidence_unit_ids=[u.unit_id for u in ordered_cited],
                evidence_sha256=sha256_hex,
                program=proposal.program,
                exe_ans=proposal.exe_ans,
                operand_sources=proposal.to_operand_sources(),
            )

        # Check 2: Temporal coherence
        if not _temporal_coherent(planner_out, proposal.operand_0, proposal.operand_1):
            return ComplianceResult(
                question_id=question_id,
                released=False,
                reason_code="ABSTAIN_TEMPORAL_MISMATCH",
                kappa=kappa,
                evidence_unit_ids=[u.unit_id for u in ordered_cited],
                evidence_sha256=sha256_hex,
                program=proposal.program,
                exe_ans=proposal.exe_ans,
                operand_sources=proposal.to_operand_sources(),
            )

        # Check 3: Joint confidence threshold kappa >= tau
        if kappa < threshold:
            return ComplianceResult(
                question_id=question_id,
                released=False,
                reason_code="ABSTAIN_LOW_CONFIDENCE",
                kappa=kappa,
                evidence_unit_ids=[u.unit_id for u in ordered_cited],
                evidence_sha256=sha256_hex,
                program=proposal.program,
                exe_ans=proposal.exe_ans,
                operand_sources=proposal.to_operand_sources(),
            )

        return ComplianceResult(
            question_id=question_id,
            released=True,
            reason_code="RELEASED",
            kappa=kappa,
            evidence_unit_ids=[u.unit_id for u in ordered_cited],
            evidence_sha256=sha256_hex,
            program=proposal.program,
            exe_ans=proposal.exe_ans,
            operand_sources=proposal.to_operand_sources(),
        )

    @staticmethod
    def write_audit_log(results: list[ComplianceResult], output_path: Path | str) -> None:
        """Write ComplianceResult entries to a JSONL audit log."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res.to_audit_dict(), ensure_ascii=False) + "\n")
