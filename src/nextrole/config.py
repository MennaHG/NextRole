"""Environment-based application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration that does not require external credentials in Phase 1."""

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


def get_settings() -> Settings:
    """Load settings from environment variables and an optional local .env file."""

    return Settings()
