"""LLM and embedding providers for the Groq-based runtime.

Groq supplies chat inference only. Embeddings are generated locally with
Hugging Face Sentence Transformers, so no OpenAI credential is required.
"""

from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_reasoning_model() -> ChatGroq:
    """Return the high-capability Groq model for analysis and auditing."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("FINANCIAL_GROQ_API_KEY is required for Groq inference.")
    return ChatGroq(
        model=settings.groq_reasoning_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


@lru_cache
def get_fast_model() -> ChatGroq:
    """Return the lower-latency Groq model for routing and decomposition."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("FINANCIAL_GROQ_API_KEY is required for Groq inference.")
    return ChatGroq(
        model=settings.groq_fast_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return local all-MiniLM embeddings; model weights are cached locally."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
