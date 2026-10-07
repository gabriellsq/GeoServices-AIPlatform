from functools import lru_cache
from pathlib import Path
from typing import Literal

from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables and `.env`.

    Credentials have no defaults: the app refuses to start without them, and SecretStr
    keeps them out of repr/str/JSON so they cannot leak into logs or tracebacks.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "geoagent"
    db_user: str
    db_password: SecretStr
    db_connect_timeout_s: int = 5
    db_connect_retries: int = 3
    db_statement_timeout_ms: int = 5000

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
            dbname=dbname or self.db_name,
            user=self.db_user,
            password=self.db_password.get_secret_value(),
            connect_timeout=self.db_connect_timeout_s,
            options=f"-c statement_timeout={timeout_ms}",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # required fields come from the environment / .env
