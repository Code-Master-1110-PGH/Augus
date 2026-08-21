"""Deterministic financial calculations exposed as LangChain tools."""

from typing import Any

from langchain_core.tools import tool


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


@tool
def revenue_growth(current_revenue: float, prior_revenue: float) -> float | None:
    """Return period-over-period revenue growth as a decimal."""
    return _safe_ratio(current_revenue - prior_revenue, prior_revenue)


@tool
def operating_margin(operating_income: float, revenue: float) -> float | None:
    """Return operating income divided by revenue."""
    return _safe_ratio(operating_income, revenue)


@tool
def debt_to_assets(total_debt: float, total_assets: float) -> float | None:
    """Return total debt divided by total assets."""
    return _safe_ratio(total_debt, total_assets)


def calculate_metrics(values: dict[str, float]) -> dict[str, Any]:
    """Calculate available metrics without asking an LLM to do arithmetic."""
    metrics: dict[str, Any] = {}
    if {"revenue", "prior_revenue"} <= values.keys():
        metrics["revenue_growth"] = revenue_growth.invoke(
            {"current_revenue": values["revenue"], "prior_revenue": values["prior_revenue"]}
        )
    if {"operating_income", "revenue"} <= values.keys():
        metrics["operating_margin"] = operating_margin.invoke(
            {"operating_income": values["operating_income"], "revenue": values["revenue"]}
        )
    if {"total_debt", "total_assets"} <= values.keys():
        metrics["debt_to_assets"] = debt_to_assets.invoke(
            {"total_debt": values["total_debt"], "total_assets": values["total_assets"]}
        )
    return metrics
