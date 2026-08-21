"""FastAPI application for autonomous financial research."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.graph import build_graph
from app.config import configure_langsmith
from app.schemas.models import ResearchRequest, ResearchStarted, ResearchStatus, ResumeRequest

configure_langsmith()
app = FastAPI(title="Financial Agentic RAG", version="0.1.0")
graph = build_graph()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def landing_page() -> FileResponse:
    """Serve the public product landing page."""
    return FileResponse(STATIC_DIR / "landing.html")


@app.get("/login", include_in_schema=False)
async def login_page() -> FileResponse:
    """Serve the sign-in gateway."""
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/workspace", include_in_schema=False)
async def workspace_page() -> FileResponse:
    """Serve the authenticated research workspace."""
    return FileResponse(STATIC_DIR / "index.html")


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


async def _run_research(thread_id: str, request: ResearchRequest) -> None:
    initial: dict[str, Any] = {
        "messages": [],
        "query": request.query,
        "company_ticker": request.company_ticker.upper(),
        "iteration_count": 0,
        "retrieved_docs": [],
        "financial_analysis": {},
        "compliance_report": {},
        "graph_history": [],
        "human_approved": False,
    }
    await graph.ainvoke(initial, _config(thread_id))


@app.post("/api/v1/research/start", response_model=ResearchStarted, status_code=202)
async def start_research(request: ResearchRequest) -> ResearchStarted:
    """Start a background research run and return its checkpoint thread ID."""
    thread_id = str(uuid.uuid4())
    asyncio.create_task(_run_research(thread_id, request))
    return ResearchStarted(thread_id=thread_id)


@app.get("/api/v1/research/{thread_id}/status", response_model=ResearchStatus)
async def research_status(thread_id: str) -> ResearchStatus:
    """Return the latest checkpoint state."""
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Research thread not found")
    state = snapshot.values
    report = state.get("compliance_report", {})
    is_human_checkpoint = bool(snapshot.next and "human_review" in snapshot.next)
    if is_human_checkpoint:
        status = "AWAITING_HUMAN_REVIEW"
    elif state.get("human_approved"):
        status = "APPROVED"
    else:
        status = report.get("status", "RUNNING")
    analysis = state.get("financial_analysis", {})
    answer = {
        "summary": analysis.get("summary", "The analyst is still processing the retrieved evidence."),
        "sentiment": analysis.get("sentiment", "pending"),
        "metrics": analysis.get("metrics", {}),
        "risk_factors": analysis.get("risk_factors", []),
        "evidence_chunk_ids": analysis.get("evidence_chunk_ids", []),
    }
    return ResearchStatus(
        thread_id=thread_id,
        question=state.get("query", ""),
        status=status,
        iteration_count=state.get("iteration_count", 0),
        compliance_report=report,
        financial_analysis=analysis,
        answer=answer,
        retrieved_documents=len(state.get("retrieved_docs", [])),
        graph_history=state.get("graph_history", []),
        human_approved=state.get("human_approved"),
    )


@app.post("/api/v1/research/{thread_id}/resume", response_model=ResearchStatus)
async def resume_research(thread_id: str, request: ResumeRequest) -> ResearchStatus:
    """Record human approval and continue from the checkpoint."""
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Research thread not found")
    graph.update_state(
        _config(thread_id),
        {"human_approved": request.approved, "graph_history": [{"node": "human_review", "reviewer": request.reviewer, "approved": request.approved, "comment": request.comment}]},
    )
    await graph.ainvoke(None, _config(thread_id))
    return await research_status(thread_id)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
