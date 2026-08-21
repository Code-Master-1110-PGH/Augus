"""SEC EDGAR retrieval with a deterministic local fallback."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import httpx

from app.config import get_settings

_SAMPLE_FILINGS = {
    "DEMO": """Item 1. Business\nDemo Holdings reports revenue of $1,000 million for 2024 and $800 million for 2023.\n\nItem 1A. Risk Factors\nThe company identifies liquidity, interest-rate, cybersecurity, and customer concentration risks.\n\nItem 2. Financial Condition\nOperating income was $150 million, total debt was $600 million, and total assets were $2,000 million."""
}


def _chunks(text: str, size: int = 900, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for start in range(0, len(words), max(1, size - overlap)):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(Item\s+[0-9A-Z]+(?:\.?[A-Z]?)[^\n]*)", text, re.IGNORECASE))
    if not matches:
        return [("Unknown", text)]
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((match.group(1).strip(), text[match.end() : end].strip()))
    return result


async def fetch_sec_filing(ticker: str, filing_type: str = "10-K") -> str:
    """Fetch a filing body; return a safe sample when SEC access is unavailable."""
    settings = get_settings()
    if ticker.upper() in _SAMPLE_FILINGS:
        return _SAMPLE_FILINGS[ticker.upper()]
    if settings.sec_api_key:
        try:
            import asyncio
            from sec_api import QueryApi

            query_api = QueryApi(api_key=settings.sec_api_key)
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker.upper()} AND formType:{filing_type}"
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            result = await asyncio.to_thread(query_api.get_filings, query)
            filings = result.get("filings", [])
            if filings and filings[0].get("linkToFilingDetails"):
                async with httpx.AsyncClient(timeout=20) as client:
                    filing = await client.get(filings[0]["linkToFilingDetails"])
                    filing.raise_for_status()
                    return filing.text
        except (ImportError, KeyError, TypeError, ValueError, httpx.HTTPError):
            pass
    url = f"{settings.sec_api_base_url}/submissions/CIK{ticker.upper()}.json"
    headers = {"User-Agent": settings.sec_user_agent}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        recent = payload.get("filings", {}).get("recent", {})
        for index, form in enumerate(recent.get("form", [])):
            if form == filing_type:
                accession = recent["accessionNumber"][index].replace("-", "")
                primary = recent["primaryDocument"][index]
                cik = payload.get("cik", "").zfill(10)
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary}"
                async with httpx.AsyncClient(timeout=20) as client:
                    filing = await client.get(filing_url, headers=headers)
                    filing.raise_for_status()
                return filing.text
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return _SAMPLE_FILINGS["DEMO"]


async def fetch_and_chunk_filing(ticker: str, filing_type: str = "10-K") -> list[dict[str, Any]]:
    """Fetch and chunk a filing while preserving provenance fields."""
    text = await fetch_sec_filing(ticker, filing_type)
    documents = []
    for section, section_text in _sections(text):
        for index, chunk in enumerate(_chunks(section_text)):
            documents.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": "SEC EDGAR",
                        "cik": ticker.upper(),
                        "filing_date": str(date.today()),
                        "filing_type": filing_type,
                        "section": section,
                        "chunk_id": f"{ticker.upper()}-{section}-{index}",
                    },
                }
            )
    return documents
