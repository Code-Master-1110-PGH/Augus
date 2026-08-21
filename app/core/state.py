"""Shared state passed through the LangGraph workflow."""

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Serializable state for one research thread."""

    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    company_ticker: str
    retrieved_docs: list[dict[str, Any]]
    financial_analysis: dict[str, Any]
    compliance_report: dict[str, Any]
    iteration_count: int
    revision_target: Literal["retrieve", "analyze"] | None
    human_approved: bool
    graph_history: list[dict[str, Any]]
