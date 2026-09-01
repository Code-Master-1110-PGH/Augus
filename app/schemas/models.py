"""Pydantic models for the research API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Input required to start a research thread."""

    query: str = Field(min_length=3, max_length=4000)
    company_ticker: str = Field(default="DEMO", min_length=1, max_length=12)


class ResearchStarted(BaseModel):
    """Acknowledgement returned for background execution."""

    thread_id: str
    status: Literal["STARTED"] = "STARTED"


class ResumeRequest(BaseModel):
    """Human decision at the compliance checkpoint."""

    approved: bool
    reviewer: str = Field(min_length=1, max_length=200)
    comment: str = Field(default="", max_length=2000)


class ResearchStatus(BaseModel):
    """Serializable current state for polling clients."""

    thread_id: str
    question: str = ""
    status: str
    iteration_count: int = 0
    compliance_report: dict[str, Any] = Field(default_factory=dict)
    financial_analysis: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    descriptive_answer: str = ""
    retrieved_documents: int = 0
    graph_history: list[dict[str, Any]] = Field(default_factory=list)
    human_approved: bool | None = None


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload and processing."""

    status: str = Field(default="success")
    message: str
    filename: str
    ticker: str
    chunks_created: int
    documents_stored: int
    chunk_ids: list[str] = Field(default_factory=list)


class DocumentUploadError(BaseModel):
    """Error response for document upload failures."""

    status: str = Field(default="error")
    message: str
    filename: str | None = None
    error_code: str = Field(default="UPLOAD_FAILED")
