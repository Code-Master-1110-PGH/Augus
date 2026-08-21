"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for API, SEC, tracing, and vector storage."""

    app_name: str = "Financial Agentic RAG"
    environment: str = "development"
    sec_user_agent: str = "FinancialAgenticRAG research@example.com"
    sec_api_base_url: str = "https://data.sec.gov"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sec_filings"
    vector_backend: str = "memory"
    openai_api_key: str = ""
    langsmith_tracing: str = "true"
    langsmith_api_key: str = ""
    langsmith_project: str = "Agentic-RAG-Financial"
    sec_api_key: str = ""
    groq_api_key: str = ""
    groq_reasoning_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_iterations: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_langsmith() -> None:
    """Expose application settings through LangSmith's standard env contract."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_tracing.lower() == "true":
        os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
