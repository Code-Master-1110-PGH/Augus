"""Evidence Parser: FinQA unit indexing, table/text normalization, and operand extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Decimal-free 4-digit integers in [1980, 2039] are treated as years, not operands.
_YEAR_RE = re.compile(r"(?<![\d.])(19[89]\d|20[0-3]\d)(?![\d.])")

# Matches currency/percent/parenthesized-negative/comma-formatted numbers.
_NUM_RE = re.compile(
    r"(?P<paren_open>\()?"
    r"(?P<sign>[+-])?\s*"
    r"(?P<currency>\$)?\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"\s*(?P<pct>%)?"
    r"(?P<paren_close>\))?"
    r"(?:\s*(?P<scale>thousand|million|billion|percent))?",
    re.IGNORECASE,
)


def extract_years(text: str) -> list[int]:
    """Return ordered unique years in [1980, 2039] (excluding decimals like 2018.5)."""
    seen: set[int] = set()
    years: list[int] = []
    for match in _YEAR_RE.finditer(text):
        yr = int(match.group(1))
        if yr not in seen:
            seen.add(yr)
            years.append(yr)
    return years


def is_year_token(raw_num: str) -> bool:
    """Return True if raw_num is a decimal-free integer in [1980, 2039]."""
    if "." in raw_num or "," in raw_num:
        return False
    try:
        val = int(raw_num)
    except ValueError:
        return False
    return 1980 <= val <= 2039


@dataclass
class EvidenceUnit:
    """A single indexed FinQA evidence unit (table_i or text_j)."""

    unit_id: str
    unit_type: str  # "table" or "text"
    index: int
    text: str
    raw_text: str
    row_label: str = ""
    headers: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    years: set[int] = field(default_factory=set)


@dataclass
class CandidateOperand:
    """A numeric operand extracted from an EvidenceUnit with provenance."""

    unit_id: str
    unit_type: str
    value: float
    raw_str: str
    is_percent: bool
    scale: str
    context: str
    row_label: str
    col_header: str
    col_index: int
    unit_years: set[int]
    col_year: int | None
    char_pos: int
    unit_rank: int = 0
    unit_score: float = 0.0
    unit_bm25: float = 0.0
    unit_tfidf: float = 0.0


class EvidenceParser:
    """Parses FinQA filings into EvidenceUnits and CandidateOperands."""

    def __init__(self, context_chars: int = 60) -> None:
        self.context_chars = context_chars

    def parse_filing(self, record: dict[str, Any]) -> list[EvidenceUnit]:
        """Parse a FinQA record into indexed EvidenceUnits preserving FinQA IDs."""
        units: list[EvidenceUnit] = []

        pre_text = record.get("pre_text") or []
        post_text = record.get("post_text") or []
        all_text = list(pre_text) + list(post_text)
        for j, sent in enumerate(all_text):
            sent_str = str(sent).strip()
            if not sent_str:
                continue
            units.append(
                EvidenceUnit(
                    unit_id=f"text_{j}",
                    unit_type="text",
                    index=j,
                    text=sent_str,
                    raw_text=sent_str,
                    years=set(extract_years(sent_str)),
                )
            )

        table = record.get("table") or []
        if table:
            headers = [str(h).strip() for h in table[0]]
            header_years = set(extract_years(" ".join(headers)))
            for i, row in enumerate(table):
                row_cells = [str(c).strip() for c in row]
                row_label = row_cells[0] if row_cells else ""
                if i == 0:
                    canonical = " | ".join(row_cells)
                else:
                    paired = [
                        f"{headers[c]}: {val}" if c < len(headers) and headers[c] else val
                        for c, val in enumerate(row_cells)
                    ]
                    canonical = " | ".join(paired)
                row_years = set(extract_years(" ".join(row_cells))) | header_years
                units.append(
                    EvidenceUnit(
                        unit_id=f"table_{i}",
                        unit_type="table",
                        index=i,
                        text=canonical,
                        raw_text=" | ".join(row_cells),
                        row_label=row_label,
                        headers=headers,
                        cells=row_cells,
                        years=row_years,
                    )
                )

        return units

    def extract_operands(self, unit: EvidenceUnit) -> list[CandidateOperand]:
        """Extract candidate numeric operands from a single EvidenceUnit."""
        if unit.unit_type == "table":
            return self._extract_table_operands(unit)
        return self._extract_text_operands(unit)

    def _parse_match(self, match: re.Match[str]) -> tuple[float, str, bool, str] | None:
        raw_num = match.group("num")
        if is_year_token(raw_num):
            return None
        try:
            val = float(raw_num.replace(",", ""))
        except ValueError:
            return None

        is_neg = bool(match.group("paren_open") and match.group("paren_close")) or (
            match.group("sign") == "-"
        )
        if is_neg:
            val = -abs(val)

        scale_raw = (match.group("scale") or "").lower()
        is_pct = bool(match.group("pct")) or scale_raw == "percent"
        if match.group("pct"):
            val = val / 100.0
        scale = scale_raw if scale_raw in {"thousand", "million", "billion"} else ""
        return val, match.group(0).strip(), is_pct, scale

    def _extract_text_operands(self, unit: EvidenceUnit) -> list[CandidateOperand]:
        text = unit.raw_text
        operands: list[CandidateOperand] = []
        for match in _NUM_RE.finditer(text):
            parsed = self._parse_match(match)
            if parsed is None:
                continue
            val, raw_str, is_pct, scale = parsed
            start = max(0, match.start() - self.context_chars)
            end = min(len(text), match.end() + self.context_chars)
            ctx = text[start:end].strip()
            ctx_years = extract_years(ctx)
            operands.append(
                CandidateOperand(
                    unit_id=unit.unit_id,
                    unit_type="text",
                    value=val,
                    raw_str=raw_str,
                    is_percent=is_pct,
                    scale=scale,
                    context=ctx,
                    row_label="",
                    col_header="",
                    col_index=0,
                    unit_years=set(unit.years),
                    col_year=ctx_years[0] if len(ctx_years) == 1 else None,
                    char_pos=match.start(),
                )
            )
        return operands

    def _extract_table_operands(self, unit: EvidenceUnit) -> list[CandidateOperand]:
        if unit.index == 0:
            # Header row itself holds labels/years, not data operands.
            return []
        operands: list[CandidateOperand] = []
        headers = unit.headers
        row_label = unit.row_label
        for c_idx, cell in enumerate(unit.cells):
            col_header = headers[c_idx] if c_idx < len(headers) else ""
            col_years = extract_years(col_header)
            col_year = col_years[0] if col_years else None
            for match in _NUM_RE.finditer(cell):
                parsed = self._parse_match(match)
                if parsed is None:
                    continue
                val, raw_str, is_pct, scale = parsed
                ctx = f"{row_label} | {col_header}: {cell}".strip(" |")
                operands.append(
                    CandidateOperand(
                        unit_id=unit.unit_id,
                        unit_type="table",
                        value=val,
                        raw_str=raw_str,
                        is_percent=is_pct,
                        scale=scale,
                        context=ctx,
                        row_label=row_label,
                        col_header=col_header,
                        col_index=c_idx,
                        unit_years=set(unit.years),
                        col_year=col_year,
                        char_pos=c_idx * 32 + match.start(),
                    )
                )
        return operands
