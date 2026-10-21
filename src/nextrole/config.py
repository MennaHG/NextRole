"""Environment-based application configuration."""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with safe, offline-first discovery defaults."""

    model_config = SettingsConfigDict(
        env_prefix="NEXTROLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile_path: Path | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "NEXTROLE_LLM_API_KEY"),
    )
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_API_KEY", "NEXTROLE_EMBEDDING_API_KEY"
        ),
    )
    log_level: str = Field(
        default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    perplexity_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PERPLEXITY_API_KEY", "NEXTROLE_PERPLEXITY_API_KEY"
        ),
    )
    max_search_requests_per_run: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices(
            "MAX_SEARCH_REQUESTS_PER_RUN", "NEXTROLE_MAX_SEARCH_REQUESTS_PER_RUN"
        ),
    )
    max_search_retries: int = Field(
        default=0,
        ge=0,
        le=1,
        validation_alias=AliasChoices(
            "MAX_SEARCH_RETRIES", "NEXTROLE_MAX_SEARCH_RETRIES"
        ),
    )
    search_max_results: int = Field(
        default=20,
        ge=1,
        le=20,
        validation_alias=AliasChoices(
            "SEARCH_MAX_RESULTS", "NEXTROLE_SEARCH_MAX_RESULTS"
        ),
    )
    search_max_age_hours: int = Field(
        default=24,
        ge=1,
        validation_alias=AliasChoices(
            "SEARCH_MAX_AGE_HOURS", "NEXTROLE_SEARCH_MAX_AGE_HOURS"
        ),
    )
    search_country: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SEARCH_COUNTRY", "NEXTROLE_SEARCH_COUNTRY"),
    )
    search_queries: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("SEARCH_QUERIES", "NEXTROLE_SEARCH_QUERIES"),
    )
    match_max_jobs: int = Field(
        default=20,
        ge=1,
        validation_alias=AliasChoices("MATCH_MAX_JOBS", "NEXTROLE_MATCH_MAX_JOBS"),
    )
    rag_chunk_size: int = Field(
        default=1200,
        ge=200,
        validation_alias=AliasChoices("RAG_CHUNK_SIZE", "NEXTROLE_RAG_CHUNK_SIZE"),
    )
    rag_chunk_overlap: int = Field(
        default=120,
        ge=0,
        validation_alias=AliasChoices(
            "RAG_CHUNK_OVERLAP", "NEXTROLE_RAG_CHUNK_OVERLAP"
        ),
    )
    rag_top_k: int = Field(
        default=4,
        ge=1,
        validation_alias=AliasChoices("RAG_TOP_K", "NEXTROLE_RAG_TOP_K"),
    )
    enrichment_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        validation_alias=AliasChoices(
            "ENRICHMENT_TIMEOUT_SECONDS", "NEXTROLE_ENRICHMENT_TIMEOUT_SECONDS"
        ),
    )
    enrichment_max_pages: int = Field(
        default=50,
        ge=1,
        validation_alias=AliasChoices(
            "ENRICHMENT_MAX_PAGES", "NEXTROLE_ENRICHMENT_MAX_PAGES"
        ),
    )
    enrichment_concurrency: int = Field(
        default=4,
        ge=1,
        validation_alias=AliasChoices(
            "ENRICHMENT_CONCURRENCY", "NEXTROLE_ENRICHMENT_CONCURRENCY"
        ),
    )
    enrichment_max_bytes: int = Field(
        default=2_000_000,
        ge=1_000,
        validation_alias=AliasChoices(
            "ENRICHMENT_MAX_BYTES", "NEXTROLE_ENRICHMENT_MAX_BYTES"
        ),
    )
    gmail_address: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GMAIL_ADDRESS", "NEXTROLE_GMAIL_ADDRESS"),
    )
    gmail_app_password: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GMAIL_APP_PASSWORD", "NEXTROLE_GMAIL_APP_PASSWORD"
        ),
    )
    digest_min_match_score: float = Field(
        default=0,
        ge=0,
        le=100,
        validation_alias=AliasChoices(
            "DIGEST_MIN_MATCH_SCORE", "NEXTROLE_DIGEST_MIN_MATCH_SCORE"
        ),
    )
    digest_max_jobs: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("DIGEST_MAX_JOBS", "NEXTROLE_DIGEST_MAX_JOBS"),
    )
    digest_state_path: Path = Field(
        default=Path("data/digest_state.json"),
        validation_alias=AliasChoices(
            "DIGEST_STATE_PATH", "NEXTROLE_DIGEST_STATE_PATH"
        ),
    )


def get_settings() -> Settings:
    """Load settings from environment variables and an optional local .env file."""

    return Settings()
