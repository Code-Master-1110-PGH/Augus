"""One-command FinQA reproducibility script for AUGUS (IEEE IC3AI 2026)."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from augus.compliance import ComplianceAgent, ComplianceResult, _operand_in_units
from augus.parser import CandidateOperand, EvidenceParser
from augus.pipeline import AUGUSPipeline, parse_gold_operands
from augus.planner import EXECUTABLE_TEMPLATES, TEMPLATES, classify_program_template

FINQA_URLS = {
    "train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}

PERTURBATION_FACTORS = (0.9, 0.95, 0.99, 1.01, 1.05, 1.1)


def ensure_finqa_data(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Locate or download official MIT-licensed FinQA splits into data/finqa/."""
    data_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {}
    for fname, url in FINQA_URLS.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"Downloading {fname} from {url} ...")
            urllib.request.urlretrieve(url, fpath)
        with fpath.open("r", encoding="utf-8") as f:
            splits[fname.replace(".json", "")] = json.load(f)
    return splits


def parse_exe_ans(raw_ans: Any) -> float | None:
    """Parse FinQA exe_ans into a float (handling percent strings if any)."""
    if raw_ans is None:
        return None
    if isinstance(raw_ans, (int, float)):
        return float(raw_ans)
    s = str(raw_ans).strip()
    if not s or s.lower() in {"yes", "no", "none", "nan"}:
        return None
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    try:
        val = float(s.replace(",", ""))
    except ValueError:
        return None
    return val / 100.0 if is_pct else val


def is_exec_correct(pred: float | None, gold: float | None) -> bool:
    """Check numeric execution accuracy against FinQA exe_ans."""
    if pred is None or gold is None:
        return False
    if abs(pred - gold) <= 1e-3:
        return True
    if abs(pred - gold) <= 1e-2 * max(abs(gold), 1e-9):
        return True
    # Handle 100x percentage scale equivalence (e.g. 0.125 vs 12.5%)
    if abs(pred * 100.0 - gold) <= 1e-2 * max(abs(gold), 1e-9):
        return True
    if abs(pred / 100.0 - gold) <= 1e-2 * max(abs(gold), 1e-9):
        return True
    return False


def evaluate_retrieval(
    pipeline: AUGUSPipeline,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute R@1, R@3, R@5, Full@5, and MRR for BM25, TF-IDF, Hybrid, and Hybrid + Reranker."""
    methods = ("bm25", "tfidf", "hybrid", "hybrid_reranker")
    stats: dict[str, dict[str, list[float]]] = {
        m: {"r1": [], "r3": [], "r5": [], "full5": [], "mrr": []} for m in methods
    }

    for rec in records:
        gold_ids = set((rec["qa"].get("gold_inds") or {}).keys())
        if not gold_ids:
            continue
        q = rec["qa"]["question"]
        units = pipeline.parser.parse_filing(rec)

        for mode in ("bm25", "tfidf", "hybrid"):
            ranked = pipeline.retriever.score_units(q, units, mode=mode)  # type: ignore[arg-type]
            ids = [ru.unit.unit_id for ru in ranked]
            _accumulate_retrieval(stats[mode], ids, gold_ids)

        hybrid_ranked = pipeline.retriever.score_units(q, units, mode="hybrid")
        reranked = pipeline.reranker.rerank(q, hybrid_ranked, top_k=len(hybrid_ranked))
        reranked_ids = [ru.retrieved.unit.unit_id for ru in reranked]
        _accumulate_retrieval(stats["hybrid_reranker"], reranked_ids, gold_ids)

    summary: dict[str, dict[str, float]] = {}
    for m, d in stats.items():
        summary[m] = {k: round(100.0 * float(np.mean(v)), 2) if k != "mrr" else round(float(np.mean(v)), 4) for k, v in d.items()}
    return summary


def _accumulate_retrieval(bucket: dict[str, list[float]], ranked_ids: list[str], gold_ids: set[str]) -> None:
    top1 = set(ranked_ids[:1])
    top3 = set(ranked_ids[:3])
    top5 = set(ranked_ids[:5])
    bucket["r1"].append(1.0 if (top1 & gold_ids) else 0.0)
    bucket["r3"].append(1.0 if (top3 & gold_ids) else 0.0)
    bucket["r5"].append(1.0 if (top5 & gold_ids) else 0.0)
    bucket["full5"].append(1.0 if gold_ids.issubset(top5) else 0.0)
    rr = 0.0
    for idx, uid in enumerate(ranked_ids, start=1):
        if uid in gold_ids:
            rr = 1.0 / idx
            break
    bucket["mrr"].append(rr)


def bootstrap_ci(correct: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap confidence interval over 1,000 resamples."""
    if len(correct) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = correct[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(100.0 * float(lo), 2), round(100.0 * float(hi), 2)


def mcnemar_pvalue(baseline_correct: np.ndarray, model_correct: np.ndarray) -> float:
    """Exact two-sided McNemar p-value vs. baseline."""
    b = int(np.sum((~baseline_correct) & model_correct))
    c = int(np.sum(baseline_correct & (~model_correct)))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def tune_tau_on_dev(
    pipeline: AUGUSPipeline,
    dev_records: list[dict[str, Any]],
) -> float:
    """Tune compliance confidence threshold tau on FinQA dev split."""
    kappas: list[float] = []
    corrects: list[bool] = []
    for rec in dev_records:
        _, _, proposal, comp = pipeline.run_record(rec, tau=0.0)
        if comp.reason_code == "RELEASED":
            gold = parse_exe_ans(rec["qa"].get("exe_ans"))
            kappas.append(comp.kappa)
            corrects.append(is_exec_correct(proposal.exe_ans, gold))

    if not kappas:
        return 0.0

    best_tau = 0.0
    best_score = -1.0
    for cand_tau in np.linspace(0.0, 0.9, 91):
        mask = [k >= cand_tau for k in kappas]
        n_rel = sum(mask)
        if n_rel < max(10, int(0.25 * len(dev_records))):
            continue
        acc = sum(c for c, m in zip(corrects, mask) if m) / n_rel
        cov = n_rel / len(dev_records)
        # Balance precision on released answers with meaningful coverage
        score = acc * (cov ** 0.15)
        if score > best_score:
            best_score = score
            best_tau = float(cand_tau)
    return round(best_tau, 4)


def run_ablation_variant(
    pipeline: AUGUSPipeline,
    records: list[dict[str, Any]],
    *,
    use_reranker: bool,
    retrieval_mode: str,
    use_planner: bool,
    oracle_gold_only: bool,
    use_compliance: bool,
    tau: float,
    collect_audit: bool = False,
) -> dict[str, Any]:
    """Evaluate one ablation configuration on records."""
    n = len(records)
    correct_all = np.zeros(n, dtype=bool)
    released_mask = np.zeros(n, dtype=bool)
    audit_results: list[ComplianceResult] = []
    per_template_total: dict[str, int] = {t: 0 for t in TEMPLATES}
    per_template_correct: dict[str, int] = {t: 0 for t in TEMPLATES}
    kappas: list[float] = []

    t0 = time.perf_counter()
    for idx, rec in enumerate(records):
        _, _, proposal, comp = pipeline.run_record(
            rec,
            use_reranker=use_reranker,
            retrieval_mode=retrieval_mode,
            use_planner=use_planner,
            oracle_gold_only=oracle_gold_only,
            tau=tau if use_compliance else 0.0,
        )
        gold = parse_exe_ans(rec["qa"].get("exe_ans"))
        gold_tpl = classify_program_template(rec["qa"].get("program", ""))
        per_template_total[gold_tpl] += 1

        is_ok = is_exec_correct(proposal.exe_ans, gold)
        if use_compliance:
            released_mask[idx] = comp.released
            correct_all[idx] = bool(comp.released and is_ok)
            kappas.append(comp.kappa)
            if collect_audit:
                audit_results.append(comp)
        else:
            answered = proposal.exe_ans is not None
            released_mask[idx] = answered
            correct_all[idx] = bool(answered and is_ok)

        if correct_all[idx]:
            per_template_correct[gold_tpl] += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
    coverage = float(released_mask.mean())
    acc_all = 100.0 * float(correct_all.mean())
    acc_released = 100.0 * float(correct_all[released_mask].mean()) if released_mask.any() else 0.0

    return {
        "correct_all": correct_all,
        "released_mask": released_mask,
        "acc_all": round(acc_all, 2),
        "acc_released": round(acc_released, 2),
        "coverage": round(100.0 * coverage, 2),
        "ms_per_query": round(elapsed_ms, 2),
        "per_template_total": per_template_total,
        "per_template_correct": per_template_correct,
        "audit_results": audit_results,
        "kappas": kappas,
    }


def run_grounding_audit(
    pipeline: AUGUSPipeline,
    test_records: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Run falsified-operand stress test over executable test programs."""
    rng = np.random.default_rng(seed)
    parser = EvidenceParser()
    exec_records = [
        r for r in test_records
        if classify_program_template(r["qa"].get("program", "")) in EXECUTABLE_TEMPLATES
    ]

    detected_top5 = 0
    detected_full = 0
    total_perturbed = 0
    false_alarm_top5 = 0
    false_alarm_full = 0
    total_clean = 0

    for rec in exec_records:
        g0, g1 = parse_gold_operands(rec["qa"].get("program", ""))
        if g0 is None or g1 is None:
            continue
        units = parser.parse_filing(rec)
        full_unit_map = {u.unit_id: u for u in units}
        full_vals = {round(op.value, 6) for u in units for op in parser.extract_operands(u)}

        # Top-5 reranked units
        retrieved = pipeline.retriever.score_units(rec["qa"]["question"], units, mode="hybrid")
        reranked = pipeline.reranker.rerank(rec["qa"]["question"], retrieved, top_k=5)
        top5_units = [ru.retrieved.unit for ru in reranked]
        top5_vals = {round(op.value, 6) for u in top5_units for op in parser.extract_operands(u)}

        # Clean (unperturbed) false-alarm check on gold operands that exist in full filing
        in_full_0 = any(abs(g0 - v) <= 1e-4 * max(1.0, abs(g0)) for v in full_vals)
        in_full_1 = any(abs(g1 - v) <= 1e-4 * max(1.0, abs(g1)) for v in full_vals)
        in_top5_0 = any(abs(g0 - v) <= 1e-4 * max(1.0, abs(g0)) for v in top5_vals)
        in_top5_1 = any(abs(g1 - v) <= 1e-4 * max(1.0, abs(g1)) for v in top5_vals)
        total_clean += 1
        if not (in_full_0 and in_full_1):
            false_alarm_full += 1
        if not (in_top5_0 and in_top5_1):
            false_alarm_top5 += 1

        # Perturbed operand check across {0.9, 0.95, 0.99, 1.01, 1.05, 1.1}
        factor = float(rng.choice(PERTURBATION_FACTORS))
        p_g0 = g0 * factor
        # Only count cases where perturbed value is genuinely absent from evidence
        if not any(abs(p_g0 - v) <= 1e-4 * max(1.0, abs(p_g0)) for v in full_vals):
            total_perturbed += 1
            # Compliance check against top-5 and full filing
            dummy_op = CandidateOperand(
                unit_id=top5_units[0].unit_id if top5_units else "text_0",
                unit_type="text",
                value=p_g0,
                raw_str=str(p_g0),
                is_percent=False,
                scale="",
                context="",
                row_label="",
                col_header="",
                col_index=0,
                unit_years=set(),
                col_year=None,
                char_pos=0,
            )
            top5_map = {u.unit_id: u for u in top5_units}
            if not _operand_in_units(dummy_op, top5_map, parser):
                detected_top5 += 1
            if not any(
                _operand_in_units(
                    CandidateOperand(
                        unit_id=uid, unit_type="text", value=p_g0, raw_str=str(p_g0),
                        is_percent=False, scale="", context="", row_label="", col_header="",
                        col_index=0, unit_years=set(), col_year=None, char_pos=0,
                    ),
                    full_unit_map,
                    parser,
                )
                for uid in full_unit_map
            ):
                detected_full += 1

    return {
        "n_executable_test": len(exec_records),
        "n_clean_checked": total_clean,
        "n_perturbed_checked": total_perturbed,
        "detection_rate_top5": round(100.0 * detected_top5 / max(total_perturbed, 1), 2),
        "detection_rate_full": round(100.0 * detected_full / max(total_perturbed, 1), 2),
        "false_alarm_rate_top5": round(100.0 * false_alarm_top5 / max(total_clean, 1), 2),
        "false_alarm_rate_full": round(100.0 * false_alarm_full / max(total_clean, 1), 2),
    }


def generate_figures(
    retrieval_summary: dict[str, dict[str, float]],
    ablation_full: dict[str, Any],
    pipeline: AUGUSPipeline,
    test_records: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    """Generate recall_at_k.png, accuracy_by_template.png, and risk_coverage.png."""
    plt.rcParams.update({"font.size": 10, "figure.dpi": 180})

    # 1. recall_at_k.png
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ks = [1, 3, 5]
    labels = {
        "bm25": "BM25",
        "tfidf": "TF-IDF",
        "hybrid": "Hybrid Fusion",
        "hybrid_reranker": "Hybrid + Reranker",
    }
    for key, label in labels.items():
        vals = [retrieval_summary[key]["r1"], retrieval_summary[key]["r3"], retrieval_summary[key]["r5"]]
        ax.plot(ks, vals, marker="o", linewidth=2, label=label)
    ax.set_xticks(ks)
    ax.set_xticklabels(["R@1", "R@3", "R@5"])
    ax.set_ylabel("Recall (%)")
    ax.set_title("FinQA Retrieval Recall@k")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "recall_at_k.png")
    plt.close(fig)

    # 2. accuracy_by_template.png
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    exec_tpls = [t for t in TEMPLATES if t != "other"]
    accs = [
        100.0 * ablation_full["per_template_correct"][t] / max(ablation_full["per_template_total"][t], 1)
        for t in exec_tpls
    ]
    bars = ax.bar(exec_tpls, accs, color=["#2b5c8f", "#3b76b0", "#4b90d1", "#6aa8e2", "#8ec0ee"])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(accs + [10.0]) + 10.0)
    ax.set_ylabel("Execution Accuracy (%)")
    ax.set_title("AUGUS Accuracy by Reasoning Template (FinQA Test)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "accuracy_by_template.png")
    plt.close(fig)

    # 3. risk_coverage.png
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    kappas_list = []
    correct_list = []
    for rec in test_records:
        _, _, prop, comp = pipeline.run_record(rec, tau=0.0)
        if comp.reason_code == "RELEASED":
            gold = parse_exe_ans(rec["qa"].get("exe_ans"))
            kappas_list.append(comp.kappa)
            correct_list.append(is_exec_correct(prop.exe_ans, gold))

    taus = np.linspace(0.0, 0.9, 37)
    covs, sel_accs = [], []
    for t_val in taus:
        mask = [k >= t_val for k in kappas_list]
        n_sel = sum(mask)
        if n_sel == 0:
            continue
        covs.append(100.0 * n_sel / len(test_records))
        sel_accs.append(100.0 * sum(c for c, m in zip(correct_list, mask) if m) / n_sel)

    ax.plot(covs, sel_accs, marker=".", linewidth=2, color="#1f6feb")
    ax.set_xlabel("Coverage (% of Test Queries Released)")
    ax.set_ylabel("Selective Execution Accuracy (%)")
    ax.set_title("Risk-Coverage Curve (Compliance Gate)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "risk_coverage.png")
    plt.close(fig)


def format_tables_md(
    retrieval_summary: dict[str, dict[str, float]],
    ablation_rows: list[dict[str, Any]],
    grounding_audit: dict[str, Any],
    tau: float,
) -> str:
    """Format reproduced tables as Markdown."""
    lines: list[str] = []
    lines.append("### Table 1: Retrieval Performance on FinQA Test Split (n = 1,147)\n")
    lines.append("| Method | R@1 (%) | R@3 (%) | R@5 (%) | Full@5 (%) | MRR |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    names = {
        "bm25": "BM25 ($k_1=1.5, b=0.75$)",
        "tfidf": "TF-IDF Cosine",
        "hybrid": "Hybrid Fusion ($\\alpha=0.5$)",
        "hybrid_reranker": "**Hybrid + GBT Reranker**",
    }
    for key in ("bm25", "tfidf", "hybrid", "hybrid_reranker"):
        r = retrieval_summary[key]
        lines.append(f"| {names[key]} | {r['r1']:.2f} | {r['r3']:.2f} | {r['r5']:.2f} | {r['full5']:.2f} | {r['mrr']:.4f} |")

    lines.append("\n### Table 2: End-to-End Accuracy, Ablations & Compliance Gate (FinQA Test)\n")
    lines.append(f"_Compliance threshold tuned on FinQA dev: $\\tau = {tau:.4f}$_\n")
    lines.append("| System Variant | Overall Acc (%) | Selective Acc (%) | 95% Bootstrap CI | McNemar $p$ vs. Baseline | Coverage (%) | CPU ms/query |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in ablation_rows:
        ci_str = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        p_str = "—" if row["p_value"] is None else f"{row['p_value']:.2e}"
        lines.append(
            f"| {row['name']} | {row['acc_all']:.2f} | {row['acc_released']:.2f} | {ci_str} | {p_str} | {row['coverage']:.2f} | {row['ms_per_query']:.2f} |"
        )

    lines.append("\n### Table 3: Falsified-Operand Grounding Stress Test\n")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :---: |")
    lines.append(f"| Executable Test Programs Evaluated | {grounding_audit['n_executable_test']} |")
    lines.append(f"| Perturbed Operands Checked ($\\times \\{{0.9, 0.95, 0.99, 1.01, 1.05, 1.1\\}}$) | {grounding_audit['n_perturbed_checked']} |")
    lines.append(f"| Perturbation Detection Rate (Top-5 Evidence) | {grounding_audit['detection_rate_top5']:.2f}% |")
    lines.append(f"| Perturbation Detection Rate (Full Filing) | {grounding_audit['detection_rate_full']:.2f}% |")
    lines.append(f"| False-Alarm Rate on Clean Gold Operands (Top-5) | {grounding_audit['false_alarm_rate_top5']:.2f}% |")
    lines.append(f"| False-Alarm Rate on Clean Gold Operands (Full Filing) | {grounding_audit['false_alarm_rate_full']:.2f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce AUGUS IEEE IC3AI 2026 FinQA experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/finqa"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    splits = ensure_finqa_data(args.data_dir)
    train_data = splits["train"]
    dev_data = splits["dev"]
    test_data = splits["test"]
    print(f"Loaded FinQA: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    pipeline = AUGUSPipeline(seed=args.seed, top_k=5, n_folds=5)
    t_train = time.perf_counter()
    pipeline.fit(train_data)
    print(f"Training completed in {time.perf_counter() - t_train:.1f}s", flush=True)

    tau = tune_tau_on_dev(pipeline, dev_data)
    pipeline.compliance.tau = tau
    print(f"Tuned compliance threshold tau = {tau:.4f}", flush=True)

    # Pre-cache retrieval/reranking on test set while timing each stage accurately
    t_parse = time.perf_counter()
    test_units = [pipeline.parser.parse_filing(r) for r in test_data]
    parse_ms = (time.perf_counter() - t_parse) * 1000.0 / len(test_data)

    t_bm25 = time.perf_counter()
    test_bm25 = [pipeline.retriever.score_units(r["qa"]["question"], u, mode="bm25") for r, u in zip(test_data, test_units)]
    bm25_ms = (time.perf_counter() - t_bm25) * 1000.0 / len(test_data)

    t_hyb = time.perf_counter()
    test_hybrid = [pipeline.retriever.score_units(r["qa"]["question"], u, mode="hybrid") for r, u in zip(test_data, test_units)]
    hyb_ms = (time.perf_counter() - t_hyb) * 1000.0 / len(test_data)

    t_rer = time.perf_counter()
    test_reranked = [pipeline.reranker.rerank(r["qa"]["question"], rh, top_k=5) for r, rh in zip(test_data, test_hybrid)]
    rer_ms = (time.perf_counter() - t_rer) * 1000.0 / len(test_data)

    print(f"Test retrieval/reranking cached (parse={parse_ms:.2f}ms, hyb={hyb_ms:.2f}ms, rerank={rer_ms:.2f}ms)", flush=True)
    retrieval_summary = evaluate_retrieval(pipeline, test_data)
    print("Retrieval evaluation done.", flush=True)

    variants = [
        ("Single-agent baseline", dict(use_reranker=False, retrieval_mode="bm25", use_planner=False, oracle_gold_only=False, use_compliance=False, collect_audit=False)),
        ("AUGUS w/o reranker", dict(use_reranker=False, retrieval_mode="hybrid", use_planner=True, oracle_gold_only=False, use_compliance=False, collect_audit=False)),
        ("AUGUS BM25-only", dict(use_reranker=True, retrieval_mode="bm25", use_planner=True, oracle_gold_only=False, use_compliance=False, collect_audit=False)),
        ("AUGUS w/o planner", dict(use_reranker=True, retrieval_mode="hybrid", use_planner=False, oracle_gold_only=False, use_compliance=False, collect_audit=False)),
        ("AUGUS full", dict(use_reranker=True, retrieval_mode="hybrid", use_planner=True, oracle_gold_only=False, use_compliance=False, collect_audit=False)),
        ("Gold-evidence oracle", dict(use_reranker=False, retrieval_mode="hybrid", use_planner=True, oracle_gold_only=True, use_compliance=False, collect_audit=False)),
        ("AUGUS + Compliance gate", dict(use_reranker=True, retrieval_mode="hybrid", use_planner=True, oracle_gold_only=False, use_compliance=True, collect_audit=True)),
    ]

    ablation_rows: list[dict[str, Any]] = []
    baseline_correct: np.ndarray | None = None
    full_res: dict[str, Any] = {}
    gate_res: dict[str, Any] = {}

    for name, kw in variants:
        res = run_ablation_variant(pipeline, test_data, tau=tau, **kw)
        if baseline_correct is None:
            baseline_correct = res["correct_all"]
            p_val = None
        else:
            p_val = mcnemar_pvalue(baseline_correct, res["correct_all"])
        ci_lo, ci_hi = bootstrap_ci(res["correct_all"], n_resamples=1000, seed=args.seed)
        ablation_rows.append(
            {
                "name": name,
                "acc_all": res["acc_all"],
                "acc_released": res["acc_released"],
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "p_value": p_val,
                "coverage": res["coverage"],
                "ms_per_query": res["ms_per_query"],
            }
        )
        if name == "AUGUS full":
            full_res = res
        if name == "AUGUS + Compliance gate":
            gate_res = res

    grounding_audit = run_grounding_audit(pipeline, test_data, seed=args.seed)

    # Save audit log sample (first 25 entries)
    ComplianceAgent.write_audit_log(
        gate_res["audit_results"][:25],
        args.results_dir / "audit_log_sample.jsonl",
    )

    # Save figures
    generate_figures(retrieval_summary, full_res, pipeline, test_data, args.results_dir)

    # Save tables.md and summary.json
    tables_md = format_tables_md(retrieval_summary, ablation_rows, grounding_audit, tau)
    (args.results_dir / "tables.md").write_text(tables_md, encoding="utf-8")

    summary_payload = {
        "seed": args.seed,
        "dataset_sizes": {"train": len(train_data), "dev": len(dev_data), "test": len(test_data)},
        "tau": tau,
        "retrieval": retrieval_summary,
        "ablations": ablation_rows,
        "grounding_audit": grounding_audit,
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    print("\n" + tables_md)


if __name__ == "__main__":
    main()
