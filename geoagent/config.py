from functools import lru_cache
from pathlib import Path
from typing import Literal

from psycopg.conninfo import make_conninfo
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"  # repo root, independent of CWD


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables and `.env`.

    Credentials have no defaults: the app refuses to start without them. SecretStr keeps
    them out of repr/str/JSON, and hide_input_in_errors keeps them out of validation errors.
    Empty values count as unset. Instances are immutable because get_settings() shares one.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        extra="ignore",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        frozen=True,
    )

    db_host: str = "127.0.0.1"
    db_port: int = Field(5432, ge=1, le=65535)
    db_name: str = "geoagent"
    db_user: str
    db_password: SecretStr
    db_connect_timeout_s: int = Field(5, ge=1)  # libpq: 0 = wait forever
    db_connect_retries: int = Field(3, ge=1)
    db_statement_timeout_ms: int = Field(5000, ge=1)

    blob_store: Literal["local", "gcs"] = "local"
    blob_root: Path = Path("blobdata")
    blob_bucket: str = "geoagent-raw"

    llm_provider: Literal["ollama", "vertex"] = "ollama"
    llm_model: str = "qwen3:8b"
    llm_timeout_s: float = 60.0
    ollama_base_url: str = "http://127.0.0.1:11434"

    gemini_api_key: SecretStr | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    embedding_location: str = "us-central1"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    top_k: int = 8
    min_similarity: float = 0.5
    log_level: str = "INFO"

    def conninfo(
        self, *, dbname: str | None = None, statement_timeout_ms: int | None = None
    ) -> str:
        """libpq connection string. It contains the password: never log it."""
        timeout_ms = (
            self.db_statement_timeout_ms if statement_timeout_ms is None else statement_timeout_ms
        )
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=dbname if dbname is not None else self.db_name,
            user=self.db_user,
            password=self.db_password.get_secret_value(),
            connect_timeout=self.db_connect_timeout_s,
            options=f"-c statement_timeout={timeout_ms}",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # required fields come from the environment / .env
