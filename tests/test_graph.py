"""Graph routing and cycle guardrail tests."""

import pytest

from app.core.graph import build_graph, route_after_audit, route_after_human_review


def test_pass_routes_to_human_review() -> None:
    assert route_after_audit({"compliance_report": {"status": "PASS"}, "iteration_count": 1}) == "narrator"


def test_retrieve_revision_routes_to_retriever() -> None:
    state = {"compliance_report": {"status": "REVISE", "revision_target": "retrieve"}, "iteration_count": 1}
    assert route_after_audit(state) == "retrieve"


def test_max_iterations_routes_to_human_review() -> None:
    state = {"compliance_report": {"status": "REVISE", "revision_target": "analyze"}, "iteration_count": 3}
    assert route_after_audit(state) == "narrator"


def test_human_approval_routes_to_end() -> None:
    assert route_after_human_review({"human_approved": True}) == "end"


def test_human_revision_routes_to_retriever() -> None:
    assert route_after_human_review({"human_approved": False, "revision_target": "retrieve"}) == "retrieve"


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid network HuggingFace downloads during offline unit tests."""
    def _raise_offline():
        raise RuntimeError("Offline test mode")

    monkeypatch.setattr("app.agents.retriever_agent.get_embeddings", _raise_offline)


@pytest.mark.asyncio
async def test_demo_graph_reaches_human_checkpoint() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "graph-test"}}
    await graph.ainvoke(
        {
            "messages": [],
            "query": "revenue growth and risk factors",
            "company_ticker": "DEMO",
            "iteration_count": 0,
            "retrieved_docs": [],
            "financial_analysis": {},
            "compliance_report": {},
            "graph_history": [],
            "human_approved": False,
        },
        config,
    )
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)


@pytest.mark.asyncio
async def test_approval_finishes_as_approved() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approval-test"}}
    initial = {
        "messages": [],
        "query": "revenue growth and risk factors",
        "company_ticker": "DEMO",
        "iteration_count": 0,
        "retrieved_docs": [],
        "financial_analysis": {},
        "compliance_report": {},
        "graph_history": [],
        "human_approved": False,
    }
    await graph.ainvoke(initial, config)
    graph.update_state(config, {"human_approved": True, "graph_history": [{"node": "human_review", "approved": True}]})
    await graph.ainvoke(None, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert snapshot.values["human_approved"] is True


@pytest.mark.asyncio
async def test_revision_returns_to_human_checkpoint() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "revision-test"}}
    initial = {
        "messages": [],
        "query": "revenue growth and risk factors",
        "company_ticker": "DEMO",
        "iteration_count": 0,
        "retrieved_docs": [],
        "financial_analysis": {},
        "compliance_report": {},
        "graph_history": [],
        "human_approved": False,
    }
    await graph.ainvoke(initial, config)
    graph.update_state(config, {"human_approved": False, "graph_history": [{"node": "human_review", "comment": "Add more evidence."}]})
    await graph.ainvoke(None, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.values["iteration_count"] == 2
