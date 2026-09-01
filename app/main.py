"""FastAPI application for autonomous financial research."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.graph import build_graph
from app.core.providers import get_embeddings
from app.config import configure_langsmith
from app.schemas.models import (
    ResearchRequest,
    ResearchStarted,
    ResearchStatus,
    ResumeRequest,
    DocumentUploadResponse,
    DocumentUploadError,
)
from app.tools.vector_store import HybridVectorStore

configure_langsmith()
app = FastAPI(title="Financial Agentic RAG", version="0.1.0")
graph = build_graph()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize vector store for document uploads
embeddings = get_embeddings()
vector_store = HybridVectorStore(embeddings=embeddings)


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
        descriptive_answer=state.get("descriptive_answer", ""),
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


@app.post("/api/v1/research/upload", response_model=DocumentUploadResponse)
async def upload_research_document(
    file: UploadFile = File(...),
    company_ticker: str = Form(default="CUSTOM", pattern=r"^[A-Z0-9.\-]{1,12}$"),
) -> DocumentUploadResponse:
    """
    Upload and process a document for RAG-based research.
    
    Supported formats: PDF, TXT, CSV
    
    Args:
        file: Document file (PDF, TXT, or CSV)
        company_ticker: Company identifier for metadata (default: CUSTOM)
        
    Returns:
        DocumentUploadResponse with processing details
        
    Raises:
        HTTPException: If file processing fails
    """
    # Validate file size (max 50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    file_size = 0
    file_content = b""
    
    try:
        # Read file in chunks to enforce size limit
        async for chunk in file.file:
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise ValueError("File exceeds maximum size of 50MB")
            file_content += chunk
        
        if not file_content:
            raise ValueError("Uploaded file is empty")
        
        # Process document
        result = await vector_store.upload_documents(
            file_bytes=file_content,
            filename=file.filename or "uploaded_document",
            company_ticker=company_ticker.upper(),
            chunk_size=512,
            overlap=100,
        )
        
        return DocumentUploadResponse(
            status="success",
            message=f"Successfully processed {file.filename}: created {result['chunks_created']} chunks",
            filename=result["filename"],
            ticker=result["ticker"],
            chunks_created=result["chunks_created"],
            documents_stored=result["documents_stored"],
            chunk_ids=result["chunk_ids"],
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}",
        )
    finally:
        await file.close()


@app.get("/api/v1/research/upload/status")
async def get_upload_status() -> dict[str, Any]:
    """
    Get upload feature status and capabilities.
    
    Returns:
        Dictionary with supported formats and limits
    """
    return {
        "status": "available",
        "supported_formats": ["PDF", "TXT", "CSV"],
        "max_file_size_mb": 50,
        "chunk_size": 512,
        "chunk_overlap": 100,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
