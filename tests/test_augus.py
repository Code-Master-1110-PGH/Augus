"""Unit tests for the 5-agent AUGUS FinQA pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augus.analyst import AnalystAgent, execute_program, operand_features
from augus.compliance import ComplianceAgent, compute_evidence_sha256
from augus.parser import CandidateOperand, EvidenceParser, EvidenceUnit, extract_years
from augus.planner import PlannerOutput, classify_program_template
from augus.reranker import reranker_features
from augus.retriever import RetrieverAgent, min_max_normalize


def test_year_filtering_excludes_integers_preserves_decimals() -> None:
    parser = EvidenceParser(context_chars=60)
    record = {
        "id": "yr-test",
        "pre_text": [
            "In 1980, 2018, and 2039 revenue was 2018.5 million and margin was 1999.0 basis points."
        ],
        "post_text": [],
        "table": [
            ["Metric", "2020", "2021"],
            ["Net income", "1,250", "1,500.25"],
        ],
    }
    units = parser.parse_filing(record)
    text_ops = parser.extract_operands(units[0])
    text_vals = [op.value for op in text_ops]

    # 1980, 2018, 2039 are decimal-free years -> excluded
    assert 1980.0 not in text_vals
    assert 2018.0 not in text_vals
    assert 2039.0 not in text_vals
    # 2018.5 and 1999.0 have decimals -> preserved
    assert 2018.5 in text_vals
    assert 1999.0 in text_vals
    assert set(extract_years(units[0].text)) == {1980, 2018, 2039}

    # Table header years (2020, 2021) should not be extracted as data operands
    table_header_ops = parser.extract_operands(units[1])
    assert table_header_ops == []
    table_row_ops = parser.extract_operands(units[2])
    assert [op.value for op in table_row_ops] == [1250.0, 1500.25]
    assert [op.col_year for op in table_row_ops] == [2020, 2021]


def test_bm25_tfidf_minmax_normalization_and_fusion_bounds() -> None:
    norm_const = min_max_normalize([5.0, 5.0, 5.0])
    assert list(norm_const) == [0.0, 0.0, 0.0]

    norm_varied = min_max_normalize([1.0, 3.0, 5.0])
    assert list(norm_varied) == [0.0, 0.5, 1.0]

    units = [
        EvidenceUnit(unit_id="text_0", unit_type="text", index=0, text="Total revenue grew to 120 million in 2022.", raw_text="Total revenue grew to 120 million in 2022."),
        EvidenceUnit(unit_id="text_1", unit_type="text", index=1, text="Operating expenses were 80 million in 2022.", raw_text="Operating expenses were 80 million in 2022."),
        EvidenceUnit(unit_id="text_2", unit_type="text", index=2, text="Unrelated footnote about office leases.", raw_text="Unrelated footnote about office leases."),
    ]
    retriever = RetrieverAgent(k1=1.5, b=0.75, alpha=0.5)
    results = retriever.score_units("What was total revenue in 2022?", units, mode="hybrid")
    assert len(results) == 3
    for ru in results:
        assert 0.0 <= ru.bm25_norm <= 1.0
        assert 0.0 <= ru.tfidf_norm <= 1.0
        assert 0.0 <= ru.hybrid_score <= 1.0
    assert results[0].unit.unit_id == "text_0"


def test_symbolic_executor_all_templates_and_zero_division() -> None:
    ans, prog, err = execute_program("ratio", 150.0, 50.0)
    assert err is None and ans == pytest.approx(3.0) and prog == "divide(150.0, 50.0)"

    ans, prog, err = execute_program("percent change", 120.0, 100.0)
    assert err is None and ans == pytest.approx(0.2) and prog == "subtract(120.0, 100.0), divide(#0, 100.0)"

    ans, prog, err = execute_program("difference", 120.0, 100.0)
    assert err is None and ans == pytest.approx(20.0) and prog == "subtract(120.0, 100.0)"

    ans, prog, err = execute_program("sum", 120.0, 100.0)
    assert err is None and ans == pytest.approx(220.0) and prog == "add(120.0, 100.0)"

    ans, prog, err = execute_program("product", 12.0, 10.0)
    assert err is None and ans == pytest.approx(120.0) and prog == "multiply(12.0, 10.0)"

    # Safe division-by-zero handling
    ans_z, _, err_z = execute_program("ratio", 100.0, 0.0)
    assert ans_z is None and err_z == "division_by_zero"

    ans_pz, _, err_pz = execute_program("percent change", 100.0, 0.0)
    assert ans_pz is None and err_pz == "division_by_zero"


def test_feature_dimensions_match_paper_spec() -> None:
    unit = EvidenceUnit(unit_id="table_1", unit_type="table", index=1, text="Revenue | 2021: 100 | 2022: 120", raw_text="Revenue | 100 | 120", row_label="Revenue", headers=["Metric", "2021", "2022"], cells=["Revenue", "100", "120"], years={2021, 2022})
    retriever = RetrieverAgent()
    ru = retriever.score_units("revenue growth 2021 2022", [unit])[0]
    r_feats = reranker_features("revenue growth 2021 2022", ru)
    assert r_feats.shape == (15,)

    cand = EvidenceParser().extract_operands(unit)[0]
    p_out = PlannerOutput(template="percent change", pi_t=0.9, other_prob=0.02, probs={"percent change": 0.9}, question_years=[2021, 2022])
    o_feats = operand_features("revenue growth 2021 2022", cand, p_out, same_unit_count=2)
    assert o_feats.shape == (26,)


def _make_cand(unit_id: str, value: float, col_year: int | None = None) -> CandidateOperand:
    return CandidateOperand(
        unit_id=unit_id,
        unit_type="table",
        value=value,
        raw_str=str(value),
        is_percent=False,
        scale="",
        context=f"Revenue | {col_year}: {value}",
        row_label="Revenue",
        col_header=str(col_year or ""),
        col_index=1,
        unit_years={col_year} if col_year else set(),
        col_year=col_year,
        char_pos=0,
    )


def test_compliance_agent_checks_and_audit_log(tmp_path: Path) -> None:
    unit_2022 = EvidenceUnit(
        unit_id="table_1",
        unit_type="table",
        index=1,
        text="Revenue | 2021: 100 | 2022: 120",
        raw_text="Revenue | 100 | 120",
        row_label="Revenue",
        headers=["Metric", "2021", "2022"],
        cells=["Revenue", "100", "120"],
        years={2021, 2022},
    )
    agent = ComplianceAgent(tau=0.5)
    planner_ok = PlannerOutput(template="percent change", pi_t=0.95, other_prob=0.01, probs={}, question_years=[2021, 2022])

    op0 = _make_cand("table_1", 120.0, col_year=2022)
    op1 = _make_cand("table_1", 100.0, col_year=2021)

    from augus.analyst import Proposal

    # 1. Valid proposal -> RELEASED
    prop_ok = Proposal(template="percent change", program="subtract(120.0, 100.0), divide(#0, 100.0)", exe_ans=0.2, operand_0=op0, operand_1=op1, p0=0.9, p1=0.9, evidence_unit_ids=["table_1"])
    res_ok = agent.verify("q-1", planner_ok, prop_ok, [unit_2022])
    assert res_ok.released is True
    assert res_ok.reason_code == "RELEASED"
    assert res_ok.evidence_sha256 == compute_evidence_sha256([unit_2022])

    # 2. Ungrounded operand -> ABSTAIN_UNGROUNDED_OPERAND
    op_fake = _make_cand("table_1", 999.0, col_year=2022)
    prop_ungrounded = Proposal(template="percent change", program="subtract(999.0, 100.0), divide(#0, 100.0)", exe_ans=8.99, operand_0=op_fake, operand_1=op1, p0=0.9, p1=0.9, evidence_unit_ids=["table_1"])
    res_ug = agent.verify("q-2", planner_ok, prop_ungrounded, [unit_2022])
    assert res_ug.released is False
    assert res_ug.reason_code == "ABSTAIN_UNGROUNDED_OPERAND"

    # 3. Temporal mismatch -> ABSTAIN_TEMPORAL_MISMATCH
    planner_2019 = PlannerOutput(template="percent change", pi_t=0.95, other_prob=0.01, probs={}, question_years=[2019, 2020])
    res_tm = agent.verify("q-3", planner_2019, prop_ok, [unit_2022])
    assert res_tm.released is False
    assert res_tm.reason_code == "ABSTAIN_TEMPORAL_MISMATCH"

    # 4. Low confidence kappa < tau -> ABSTAIN_LOW_CONFIDENCE
    prop_low = Proposal(template="percent change", program="subtract(120.0, 100.0), divide(#0, 100.0)", exe_ans=0.2, operand_0=op0, operand_1=op1, p0=0.4, p1=0.4, evidence_unit_ids=["table_1"])
    res_lc = agent.verify("q-4", planner_ok, prop_low, [unit_2022])
    assert res_lc.released is False
    assert res_lc.reason_code == "ABSTAIN_LOW_CONFIDENCE"

    # 5. JSONL audit log deterministic SHA-256
    log_path = tmp_path / "audit.jsonl"
    ComplianceAgent.write_audit_log([res_ok, res_ug], log_path)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["question_id"] == "q-1"
    assert row0["evidence_sha256"] == compute_evidence_sha256([unit_2022])
    assert row0["released"] is True


def test_falsified_operand_perturbation_100_percent_detection() -> None:
    unit = EvidenceUnit(
        unit_id="text_0",
        unit_type="text",
        index=0,
        text="Operating income was 250.0 million and revenue was 1000.0 million.",
        raw_text="Operating income was 250.0 million and revenue was 1000.0 million.",
    )
    agent = ComplianceAgent(tau=0.0)
    planner = PlannerOutput(template="ratio", pi_t=1.0, other_prob=0.0, probs={}, question_years=[])
    op1 = _make_cand("text_0", 1000.0)

    from augus.analyst import Proposal

    for factor in (0.9, 0.95, 0.99, 1.01, 1.05, 1.1):
        perturbed_val = 250.0 * factor
        op0_perturbed = _make_cand("text_0", perturbed_val)
        prop = Proposal(
            template="ratio",
            program=f"divide({perturbed_val}, 1000.0)",
            exe_ans=perturbed_val / 1000.0,
            operand_0=op0_perturbed,
            operand_1=op1,
            p0=1.0,
            p1=1.0,
            evidence_unit_ids=["text_0"],
        )
        res = agent.verify("perturb", planner, prop, [unit])
        assert res.released is False
        assert res.reason_code == "ABSTAIN_UNGROUNDED_OPERAND"


def test_program_template_classification() -> None:
    assert classify_program_template("divide(100, 50)") == "ratio"
    assert classify_program_template("subtract(120, 100), divide(#0, 100)") == "percent change"
    assert classify_program_template("subtract(120, 100)") == "difference"
    assert classify_program_template("add(120, 100)") == "sum"
    assert classify_program_template("multiply(12, 10)") == "product"
    assert classify_program_template("divide(100, const_100)") == "other"
    assert classify_program_template("add(1, 2), divide(#0, 3)") == "other"
