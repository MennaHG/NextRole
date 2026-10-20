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
    embedding_provider: str | None = None
    embedding_model: str | None = None
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


def get_settings() -> Settings:
    """Load settings from environment variables and an optional local .env file."""

    return Settings()
