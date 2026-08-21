# Financial Agentic RAG

Production-oriented reference implementation for autonomous SEC financial research with LangGraph, FastAPI, provenance-preserving retrieval, deterministic calculations, compliance routing, and human approval checkpoints.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The default `DEMO` ticker uses an offline SEC filing fixture, so the workflow and tests run without API keys. Other tickers attempt SEC EDGAR using `FINANCIAL_SEC_USER_AGENT` and fall back to the same safe fixture if the network is unavailable. Set `FINANCIAL_VECTOR_BACKEND=qdrant` and configure the Qdrant URL when replacing the in-memory adapter.

## Groq and local embeddings

The provider boundary in `app/core/providers.py` uses `ChatGroq` from
`langchain-groq` for LLM inference and `HuggingFaceEmbeddings` from
`langchain-huggingface` for local `sentence-transformers/all-MiniLM-L6-v2`
embeddings. No OpenAI API key or OpenAI embedding model is required.

```powershell
pip install langchain-groq langchain-huggingface sentence-transformers sec-api
$env:GROQ_API_KEY="gsk_..."
$env:SEC_API_KEY="..."
```

The default Groq models are `llama-3.3-70b-versatile` for complex reasoning
and `llama-3.1-8b-instant` for low-latency routing. Override them with
`GROQ_REASONING_MODEL` and `GROQ_FAST_MODEL`.

The existing financial calculations and compliance rules remain deterministic.
This is intentional: arithmetic and hard compliance gates should not be
delegated to an LLM. Call `get_reasoning_model()` or `get_fast_model()` from
an agent when adding qualitative LLM reasoning to a node.

To enable LangSmith tracing, put your LangSmith token in the ignored `.env`
file as `LANGSMITH_API_KEY=...` and set
`LANGSMITH_TRACING=true`. The application maps these settings to
LangSmith's standard environment variables at startup. Never commit `.env`
or place the token in Python, frontend, Docker images, or README files.

## API workflow

1. `POST /api/v1/research/start` with `{"query":"revenue growth and risks","company_ticker":"DEMO"}`.
2. Poll `GET /api/v1/research/{thread_id}/status` until `AWAITING_HUMAN_REVIEW`.
3. Submit `POST /api/v1/research/{thread_id}/resume` with reviewer approval.

LangGraph `MemorySaver` is used by default. For multi-process production deployments, replace it with a Postgres checkpointer and put the API behind a durable task queue. LangSmith tracing is enabled through its standard environment variables when configured.

## Tests and evaluation

```powershell
pytest
python -m app.evaluation.ragas_eval
```

The evaluation runner emits reproducible local proxies for context precision, faithfulness, and answer relevance. A RAGAS-backed evaluator can consume the same retrieved context and answer dataset in CI.
