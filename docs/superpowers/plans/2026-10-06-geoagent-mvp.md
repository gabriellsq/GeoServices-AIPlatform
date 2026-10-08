# GeoAgent MVP (Sprint 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Two-zone rule (spec §8) — read before executing any task.** Every task is tagged:
> - **[LLM zone]** — the assistant implements; the developer reviews the diff before commit. Full code is in this plan.
> - **[Developer zone]** — the assistant writes the failing tests (full code in this plan) and reviews; **the developer writes the implementation**. This plan deliberately gives only the *contract* (signatures, behaviour, hints), never the implementation. An executing agent must stop after writing the tests and hand over to the developer.

**Goal:** A RAG service over real NZ NI 43-101 geology reports that runs locally (Docker Compose + Ollama + Gemini embeddings) and deploys once to GCP (Cloud Run + Cloud SQL/pgvector + GCS + Vertex AI) via Terraform.

**Architecture:** Plain Python, no LLM framework. Ingestion (parse → chunk → embed → store) and serving (embed question → pgvector top-k → grounded prompt → generate → cited answer) share one package `geoagent` and one container image. Providers (LLM, embeddings, blob store) sit behind small protocols selected by environment variables, so local and cloud differ only in configuration.

**Tech Stack:** Python 3.12, uv, FastAPI, Typer, psycopg 3, pgvector, PyMuPDF, tiktoken, google-genai, google-cloud-storage, httpx, pytest, ruff, Docker Compose, Ollama, Terraform (google provider), Cloud Run, Cloud SQL for PostgreSQL 16, GCS, Secret Manager, Vertex AI.

**Spec:** `docs/superpowers/specs/2026-10-06-geoagent-mvp-design.md`

---

## Conventions used by every task

- Run all commands from the repo root.
- Unit tests: `uv run pytest -m "not integration"`. Integration tests need Postgres: `docker compose up -d postgres` first, then `uv run pytest -m integration`.
- Database connections are opened with `autocommit=True`; atomic units of work use `with conn.transaction():`.
- Local Postgres is always addressed as `127.0.0.1` (never `localhost`: on Windows, WSL's `wslrelay` can hold `[::1]:5432` and black-hole IPv6 connections) with a `connect_timeout`, so a stopped container fails fast.
- Everything is **synchronous** (sync psycopg, sync httpx, sync FastAPI endpoints run in the threadpool). Async is a later-sprint topic.
- Commits: conventional-commit subject, no AI attribution trailer. Before each commit, check staged files contain no personal information.

## File map

```
GeoSolution/
├── pyproject.toml, uv.lock, .env.example, .gitignore, .dockerignore, README.md
├── Dockerfile, docker-compose.yml
├── geoagent/
│   ├── __init__.py
│   ├── config.py                  [LLM]   Settings
│   ├── logs.py                    [LLM]   JSON logging + request_id contextvar
│   ├── wiring.py                  [LLM]   build providers from Settings
│   ├── smoke.py                   [LLM]   smoke-set loader
│   ├── cli.py                     [LLM]   Typer CLI
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py          [LLM]   connect() + register_vector
│   │   ├── migrate.py             [LLM]   migration runner
│   │   └── migrations/001_init.sql [DEV]  schema
│   ├── blobstore/
│   │   ├── __init__.py
│   │   ├── base.py                [LLM]   BlobStore protocol
│   │   ├── local.py               [LLM]   LocalFsBlobStore
│   │   └── gcs.py                 [LLM]   GcsBlobStore
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── types.py               [DEV]   TaskType, Generation, LLMProvider, EmbeddingProvider
│   │   ├── errors.py              [DEV]   ProviderError hierarchy
│   │   ├── ollama.py              [DEV]   OllamaProvider
│   │   └── gemini.py              [LLM]   GeminiEmbeddings, VertexGeminiProvider
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── parse.py               [DEV]   PDF → pages
│   │   ├── chunker.py             [DEV]   pages → chunks
│   │   └── pipeline.py            [DEV]   orchestration + status
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py           [DEV]   pgvector top-k
│   │   ├── prompt.py              [DEV]   grounded prompt + citation parsing
│   │   └── answer.py              [DEV]   end-to-end answer
│   └── api/
│       ├── __init__.py
│       ├── schemas.py             [LLM]
│       └── main.py                [LLM]   FastAPI app
├── scripts/
│   ├── fetch_reports.py           [LLM]
│   └── compare_embeddings.py      [LLM]
├── tests/
│   ├── __init__.py, conftest.py   [LLM]
│   ├── fakes.py                   [LLM]   FakeEmbedder, FakeLLM, vectors
│   ├── fixtures/__init__.py, fixture_pdf.py [LLM]  synthetic PDF generator
│   ├── unit/…                     [LLM writes all tests]
│   └── integration/…              [LLM writes all tests]
├── evals/smoke.jsonl              [DEV]   ~30 hand-written questions
├── infra/terraform/
│   ├── bootstrap/main.tf          [DEV]
│   └── main/                      [LLM: versions/variables/outputs/locals] [DEV: resource files]
└── docs/
    ├── learning-log.md            [DEV entries]
    ├── local-llm.md               [LLM]
    └── deploy.md                  [LLM]
```

---

## Task 1: Project scaffold [LLM zone]

> **Status: done.** Implemented with review amendments; the committed files are the source of truth where they differ from the text below (credentials moved to `.env` as split `DB_*` fields, `127.0.0.1` everywhere, `SecretStr`, timeout budget, settings hardening).

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.dockerignore`, `README.md`, `docs/learning-log.md`
- Create: `geoagent/__init__.py`, `geoagent/db/__init__.py`, `geoagent/blobstore/__init__.py`, `geoagent/providers/__init__.py`, `geoagent/ingest/__init__.py`, `geoagent/rag/__init__.py`, `geoagent/api/__init__.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/fixtures/__init__.py`, `tests/unit/test_smoke_import.py`
- Modify: `.gitignore`

- [ ] **Step 1: Install uv if missing**

Run: `uv --version`
Expected: a version string. If not found, install from https://docs.astral.sh/uv/getting-started/installation/ and re-run.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "geoagent"
version = "0.1.0"
description = "RAG over geoscience technical reports - a production-style GenAI learning project"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
    "pydantic-settings>=2.4",
    "typer>=0.12",
    "httpx>=0.27",
    "psycopg[binary]>=3.2",
    "pgvector>=0.3",
    "numpy>=1.26",
    "pymupdf>=1.24",
    "tiktoken>=0.7",
    "google-genai>=1.0",
    "google-cloud-storage>=2.18",
]

[project.scripts]
geoagent = "geoagent.cli:app"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["geoagent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["integration: needs `docker compose up -d postgres`"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]  # line length is handled by `ruff format`
```

- [ ] **Step 3: Create empty package/test `__init__.py` files**

Create each `__init__.py` listed under **Files** as an empty file.

- [ ] **Step 4: Create `.env.example`**

```dotenv
# Copy to .env and fill in. .env is git-ignored.
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=geoagent
DB_USER=
DB_PASSWORD=

BLOB_STORE=local
BLOB_ROOT=blobdata
BLOB_BUCKET=geoagent-raw

LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://gpu-host.lan:11434

# AI Studio key, used for embeddings locally (https://aistudio.google.com/apikey)
GEMINI_API_KEY=
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=768
EMBEDDING_BATCH_SIZE=32

TOP_K=8
MIN_SIMILARITY=0.5
LOG_LEVEL=INFO
```

- [ ] **Step 5: Create `.dockerignore`**

```
.git
.venv
data
blobdata
docs
tests
infra
evals
**/__pycache__
.env
```

- [ ] **Step 6: Append to `.gitignore`**

Append these lines to the existing `.gitignore`:

```
# Local blob store
blobdata/
```

- [ ] **Step 7: Create `README.md`**

```markdown
# GeoAgent

Retrieval-augmented generation over public geoscience technical reports (NI 43-101), built as a
production-style GenAI learning project: plain Python, Postgres + pgvector, Gemini embeddings,
Ollama locally, Vertex AI + Cloud Run + Cloud SQL in the cloud, Terraform for infrastructure.

- Design: `docs/superpowers/specs/2026-10-06-geoagent-mvp-design.md`
- Plan: `docs/superpowers/plans/2026-10-06-geoagent-mvp.md`
- Learning log: `docs/learning-log.md`

Source reports are copyrighted and are not stored in this repository; `scripts/fetch_reports.py`
downloads them into the git-ignored `data/` directory.
```

- [ ] **Step 8: Create `docs/learning-log.md`**

```markdown
# Learning log

One entry per experiment. Write the prediction **before** running anything.

## Template

### YYYY-MM-DD — <experiment name>
- **Hypothesis / prediction:**
- **Setup (what changed, what stayed fixed):**
- **Result (numbers, examples):**
- **Why (explanation in my own words):**
- **Key takeaway (one sentence):**
```

- [ ] **Step 9: Write a smoke import test**

`tests/unit/test_smoke_import.py`:

```python
import geoagent


def test_package_imports():
    assert geoagent.__name__ == "geoagent"
```

- [ ] **Step 10: Sync and run**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`, installs all dependencies.

Run: `uv run pytest -q`
Expected: `1 passed`.

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml uv.lock .env.example .dockerignore .gitignore README.md docs/learning-log.md geoagent tests
git commit -m "chore: scaffold geoagent package, tooling and learning log"
```

---

## Task 2: Docker Compose Postgres + pgvector [LLM zone]

> **Status: done.** Implemented with review amendments; the committed files are the source of truth where they differ from the text below (credentials moved to `.env` as split `DB_*` fields, `127.0.0.1` everywhere, `SecretStr`, timeout budget, settings hardening).

**Files:**
- Create: `docker-compose.yml`
- Create: `tests/integration/test_postgres_available.py`

- [ ] **Step 1: Create `docker-compose.yml`** (the `api` service is added in Task 18)

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${DB_USER:?set DB_USER in .env}
      POSTGRES_PASSWORD: ${DB_PASSWORD:?set DB_PASSWORD in .env}
      POSTGRES_DB: ${DB_NAME:-geoagent}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 20

volumes:
  pgdata:
```

- [ ] **Step 2: Write the availability test**

`tests/integration/test_postgres_available.py`:

```python
import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_pgvector_extension_is_available():
    with psycopg.connect(Settings().conninfo()) as conn:
        row = conn.execute(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
        ).fetchone()
    assert row is not None, "pgvector is not installed in this Postgres image"
```

- [ ] **Step 3: Run it with Postgres down**

Run: `uv run pytest -m integration -q`
Expected: FAIL with `psycopg.OperationalError` (connection refused).

- [ ] **Step 4: Start Postgres and re-run**

Run: `docker compose up -d postgres` then `docker compose ps`
Expected: `postgres` is `healthy` (re-run `docker compose ps` until it is).

Run: `uv run pytest -m integration -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml tests/integration/test_postgres_available.py
git commit -m "chore: add postgres+pgvector compose service"
```

---

## Task 3: Settings [LLM zone]

> **Status: done.** Implemented with review amendments; the committed files are the source of truth where they differ from the text below (credentials moved to `.env` as split `DB_*` fields, `127.0.0.1` everywhere, `SecretStr`, timeout budget, settings hardening).

**Files:**
- Create: `geoagent/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:

```python
from geoagent.config import Settings


def test_defaults_target_local_development(monkeypatch):
    for var in ["LLM_PROVIDER", "BLOB_STORE", "EMBEDDING_DIM", "TOP_K"]:
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.blob_store == "local"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.embedding_dim == 768
    assert s.embedding_batch_size == 32
    assert s.top_k == 8


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("BLOB_STORE", "gcs")
    monkeypatch.setenv("TOP_K", "3")
    s = Settings(_env_file=None)
    assert s.llm_provider == "vertex"
    assert s.blob_store == "gcs"
    assert s.top_k == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.config'`.

- [ ] **Step 3: Implement `geoagent/config.py`**

```python
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # see committed geoagent/config.py: split DB_* fields, db_password: SecretStr (required)

    blob_store: Literal["local", "gcs"] = "local"
    blob_root: Path = Path("blobdata")
    blob_bucket: str = "geoagent-raw"

    llm_provider: Literal["ollama", "vertex"] = "ollama"
    llm_model: str = "qwen3:14b"
    llm_timeout_s: float = 60.0
    ollama_base_url: str = "http://127.0.0.1:11434"

    gemini_api_key: str | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    embedding_location: str = "us-central1"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    top_k: int = 8
    min_similarity: float = 0.5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_config.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add geoagent/config.py tests/unit/test_config.py
git commit -m "feat: add typed settings"
```

---

## Task 4: Schema migration [SQL: Developer zone; runner + tests: LLM zone]

**Files:**
- Create (LLM): `geoagent/db/connection.py`, `geoagent/db/migrate.py`, `geoagent/db/workspaces.py`, `tests/unit/test_db_connection.py`, `tests/integration/conftest.py`, `tests/integration/test_migrations.py`, `tests/integration/test_workspaces.py`
- Create: `geoagent/db/migrations/001_init.sql` (developer chose subagent mode for this task)

**Contract for `001_init.sql`:** exactly the schema below (also spec §6, schema v2: UUID workspace key + slug, composite tenant foreign key).

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Tenant. UUID primary key (meaningless, immutable); slug is the human-readable handle.
CREATE TABLE workspaces (
    id         UUID PRIMARY KEY,
    slug       TEXT NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per source PDF. No ON DELETE on the workspace FK: tenant offboarding is explicit.
CREATE TABLE documents (
    id           UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    title        TEXT NOT NULL,
    source_url   TEXT,
    blob_uri     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('uploaded','processing','ready','failed')),
    error        TEXT,
    page_count   INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, sha256),
    UNIQUE (id, workspace_id)            -- target of the composite FK below
);

-- One row per chunk. workspace_id is denormalised for filtering; the composite FK
-- guarantees it always equals the parent document's workspace.
CREATE TABLE chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL,
    workspace_id UUID NOT NULL,
    ordinal      INT  NOT NULL,
    page_start   INT  NOT NULL,
    page_end     INT  NOT NULL,
    section      TEXT,
    text         TEXT NOT NULL,
    token_count  INT  NOT NULL,
    embedding    vector(768) NOT NULL,
    UNIQUE (document_id, ordinal),
    FOREIGN KEY (document_id, workspace_id)
        REFERENCES documents (id, workspace_id) ON DELETE CASCADE
);

CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_workspace ON chunks (workspace_id);
```

- [ ] **Step 1a [LLM]: Write the failing connection tests**

`tests/unit/test_db_connection.py`:

```python
import logging

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from geoagent.config import Settings
from geoagent.db import connection

PASSWORD = "hunter2-not-real"


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, db_user="app", db_password=PASSWORD, **overrides)


def test_retries_with_backoff_then_succeeds(monkeypatch):
    calls, sleeps, sentinel = [], [], object()

    def fake_connect(conninfo, autocommit):
        calls.append(conninfo)
        if len(calls) < 3:
            raise psycopg.OperationalError("connection refused")
        return sentinel

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    conn = connection.connect(
        settings(db_connect_retries=3), register_vector_type=False, sleep=sleeps.append
    )
    assert conn is sentinel
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] < 0.75 and 1.0 <= sleeps[1] < 1.25


def test_gives_up_after_configured_attempts(monkeypatch):
    attempts = []

    def fake_connect(conninfo, autocommit):
        attempts.append(1)
        raise psycopg.OperationalError("down")

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    with pytest.raises(psycopg.OperationalError):
        connection.connect(
            settings(db_connect_retries=2), register_vector_type=False, sleep=lambda s: None
        )
    assert len(attempts) == 2


def test_passes_dbname_statement_timeout_and_autocommit(monkeypatch):
    seen = {}

    def fake_connect(conninfo, autocommit):
        seen.update(conninfo=conninfo, autocommit=autocommit)
        return object()

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    connection.connect(
        settings(), dbname="other", statement_timeout_ms=120000, register_vector_type=False
    )
    params = conninfo_to_dict(seen["conninfo"])
    assert params["dbname"] == "other"
    assert params["options"] == "-c statement_timeout=120000"
    assert seen["autocommit"] is True


def test_retry_logs_never_contain_the_password(monkeypatch, caplog):
    def fake_connect(conninfo, autocommit):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    with caplog.at_level(logging.DEBUG), pytest.raises(psycopg.OperationalError):
        connection.connect(settings(), register_vector_type=False, sleep=lambda s: None)
    assert caplog.records, "retries should be logged"
    assert PASSWORD not in caplog.text
```

Run: `uv run pytest tests/unit/test_db_connection.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.db.connection'`.

- [ ] **Step 1b [LLM]: Implement `geoagent/db/connection.py`**

```python
import logging
import random
import time
from collections.abc import Callable

import psycopg
from pgvector.psycopg import register_vector

from geoagent.config import Settings

log = logging.getLogger(__name__)


def connect(
    settings: Settings,
    *,
    dbname: str | None = None,
    statement_timeout_ms: int | None = None,
    register_vector_type: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> psycopg.Connection:
    """Open an autocommit connection, retrying transient failures with jittered backoff.

    `settings.db_connect_retries` is the total number of attempts. The conninfo string
    contains the password, so it is never logged. Set `register_vector_type=False` when the
    `vector` extension may not exist yet (migrations, admin connections).
    """
    conninfo = settings.conninfo(dbname=dbname, statement_timeout_ms=statement_timeout_ms)
    attempts = settings.db_connect_retries
    for attempt in range(1, attempts + 1):
        try:
            conn = psycopg.connect(conninfo, autocommit=True)
            break
        except psycopg.OperationalError as exc:
            if attempt == attempts:
                raise
            delay = 0.5 * 2 ** (attempt - 1) + random.uniform(0, 0.25)
            log.warning(
                "database connection failed, retrying",
                extra={
                    "attempt": attempt,
                    "retry_in_s": round(delay, 2),
                    "error": type(exc).__name__,
                },
            )
            sleep(delay)
    if register_vector_type:
        register_vector(conn)
    return conn
```

Run: `uv run pytest tests/unit/test_db_connection.py -q`
Expected: `4 passed`.

- [ ] **Step 2 [LLM]: Implement `geoagent/db/migrate.py`**

```python
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every *.sql file not yet recorded in schema_migrations, in filename order.

    Each file runs in its own transaction together with its bookkeeping row.
    Returns the names of the files applied by this call.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name TEXT PRIMARY KEY,"
        " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
    applied: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in done:
            continue
        with conn.transaction():
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
        applied.append(path.name)
    return applied
```

- [ ] **Step 3 [LLM]: Create `tests/integration/conftest.py`** (shared by all integration tests)

```python
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from geoagent.config import Settings
from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations


def recreate_database(settings: Settings, name: str) -> None:
    with connect(settings, dbname="postgres", register_vector_type=False) as admin:
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Credentials come from the repo `.env` (the same file docker compose reads)."""
    return Settings()


@pytest.fixture(scope="session")
def test_db(settings: Settings) -> str:
    name = f"{settings.db_name}_test"
    recreate_database(settings, name)
    with connect(settings, dbname=name, register_vector_type=False) as c:
        apply_migrations(c)
    return name


@pytest.fixture
def conn(settings: Settings, test_db: str) -> Iterator[psycopg.Connection]:
    c = connect(settings, dbname=test_db)
    c.execute("TRUNCATE chunks, documents, workspaces CASCADE")
    yield c
    c.close()
```

- [ ] **Step 4 [LLM]: Write the failing schema tests**

`tests/integration/test_migrations.py`:

```python
import uuid

import psycopg
import pytest

from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations
from tests.integration.conftest import recreate_database

pytestmark = pytest.mark.integration

MIGRATION_DB = "geoagent_migration_test"


@pytest.fixture
def fresh(settings) -> psycopg.Connection:
    recreate_database(settings, MIGRATION_DB)
    with connect(settings, dbname=MIGRATION_DB, register_vector_type=False) as c:
        yield c


def test_applies_once_and_is_idempotent(fresh):
    assert apply_migrations(fresh) == ["001_init.sql"]
    assert apply_migrations(fresh) == []


def test_embedding_column_is_vector_768(fresh):
    apply_migrations(fresh)
    row = fresh.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()
    assert row == ("vector(768)",)


def test_hnsw_cosine_index_exists(fresh):
    apply_migrations(fresh)
    row = fresh.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
    ).fetchone()
    assert row is not None
    assert "hnsw" in row[0] and "vector_cosine_ops" in row[0]


def make_workspace(conn, slug: str = "ws") -> uuid.UUID:
    workspace_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO workspaces (id, slug, name) VALUES (%s, %s, %s)",
        (workspace_id, slug, slug.upper()),
    )
    return workspace_id


def make_document(conn, workspace_id, *, sha: str = "h", status: str = "ready") -> uuid.UUID:
    document_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO documents (id, workspace_id, title, blob_uri, sha256, status) "
        "VALUES (%s, %s, 't', 'local://b/k', %s, %s)",
        (document_id, workspace_id, sha, status),
    )
    return document_id


def make_chunk(conn, document_id, workspace_id) -> None:
    conn.execute(
        "INSERT INTO chunks (id, document_id, workspace_id, ordinal, page_start, page_end, "
        "text, token_count, embedding) "
        "VALUES (%s, %s, %s, 0, 1, 1, 'x', 1, array_fill(0.1, ARRAY[768])::vector)",
        (uuid.uuid4(), document_id, workspace_id),
    )


def test_status_check_constraint(fresh):
    apply_migrations(fresh)
    ws = make_workspace(fresh)
    with pytest.raises(psycopg.errors.CheckViolation):
        make_document(fresh, ws, status="bogus")


def test_sha256_unique_per_workspace_only(fresh):
    apply_migrations(fresh)
    ws_a, ws_b = make_workspace(fresh, "ws-a"), make_workspace(fresh, "ws-b")
    make_document(fresh, ws_a, sha="same")
    make_document(fresh, ws_b, sha="same")  # same PDF in another tenant is allowed
    with pytest.raises(psycopg.errors.UniqueViolation):
        make_document(fresh, ws_a, sha="same")


def test_deleting_document_cascades_to_chunks(fresh):
    apply_migrations(fresh)
    ws = make_workspace(fresh)
    doc = make_document(fresh, ws)
    make_chunk(fresh, doc, ws)
    fresh.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert fresh.execute("SELECT count(*) FROM chunks").fetchone() == (0,)


def test_chunk_cannot_belong_to_another_tenant_than_its_document(fresh):
    apply_migrations(fresh)
    ws_a, ws_b = make_workspace(fresh, "ws-a"), make_workspace(fresh, "ws-b")
    doc_a = make_document(fresh, ws_a)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        make_chunk(fresh, doc_a, ws_b)


def test_workspace_with_documents_cannot_be_deleted(fresh):
    apply_migrations(fresh)
    ws = make_workspace(fresh)
    make_document(fresh, ws)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        fresh.execute("DELETE FROM workspaces WHERE id = %s", (ws,))


def test_workspace_slug_is_unique(fresh):
    apply_migrations(fresh)
    make_workspace(fresh, "tenant-a")
    with pytest.raises(psycopg.errors.UniqueViolation):
        make_workspace(fresh, "tenant-a")


@pytest.mark.parametrize("bad_slug", ["Tenant A", "a", "-leading-dash", "under_score"])
def test_workspace_slug_format_is_enforced(fresh, bad_slug):
    apply_migrations(fresh)
    with pytest.raises(psycopg.errors.CheckViolation):
        make_workspace(fresh, bad_slug)
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/integration/test_migrations.py -q`
Expected: FAIL — `apply_migrations` returns `[]` (no SQL file yet) and table lookups raise `UndefinedTable`.

- [ ] **Step 6: Write `geoagent/db/migrations/001_init.sql`** (exactly the contract SQL above)

Things to understand (developer review):
- `vector(768)` fixes dimensionality at the type level — inserting a 767-dim vector fails. Why 768 and not 3072? (pgvector HNSW limit: 2,000 dims for `vector`.)
- `vector_cosine_ops` must match the operator used at query time (`<=>`), otherwise the index is ignored.
- Why is `workspace_id` duplicated on `chunks` when it is derivable via `documents`, and how does the composite foreign key make that safe?
- Why a UUID primary key plus a `slug`, instead of the slug as the key?

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/integration/test_migrations.py -q`
Expected: `13 passed`.

- [ ] **Step 7b [LLM]: Workspace helpers (resolve slug ↔ UUID at the edges)**

`tests/integration/test_workspaces.py`:

```python
import uuid

import psycopg
import pytest

from geoagent.db.workspaces import WorkspaceNotFound, create_workspace, get_workspace_id

pytestmark = pytest.mark.integration


def test_create_then_resolve_by_slug(conn):
    workspace_id = create_workspace(conn, slug="tenant-a", name="Tenant A")
    assert isinstance(workspace_id, uuid.UUID)
    assert get_workspace_id(conn, "tenant-a") == workspace_id


def test_unknown_slug_raises(conn):
    with pytest.raises(WorkspaceNotFound):
        get_workspace_id(conn, "nobody")


def test_duplicate_slug_is_rejected(conn):
    create_workspace(conn, slug="tenant-a", name="Tenant A")
    with pytest.raises(psycopg.errors.UniqueViolation):
        create_workspace(conn, slug="tenant-a", name="Again")
```

Run: `uv run pytest tests/integration/test_workspaces.py -q` → FAIL (`ModuleNotFoundError`).

`geoagent/db/workspaces.py`:

```python
import uuid

import psycopg


class WorkspaceNotFound(LookupError):
    """No workspace has this slug."""


def create_workspace(conn: psycopg.Connection, *, slug: str, name: str) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO workspaces (id, slug, name) VALUES (%s, %s, %s)",
        (workspace_id, slug, name),
    )
    return workspace_id


def get_workspace_id(conn: psycopg.Connection, slug: str) -> uuid.UUID:
    row = conn.execute("SELECT id FROM workspaces WHERE slug = %s", (slug,)).fetchone()
    if row is None:
        raise WorkspaceNotFound(slug)
    return row[0]
```

Run: `uv run pytest tests/integration/test_workspaces.py -q` → `3 passed`.

- [ ] **Step 8 [LLM]: Review questions for the developer**

Ask (do not answer): What happens to an HNSW index build time as rows grow? What does `ON DELETE CASCADE` mean for the "deleted document still retrievable" problem? What would a second migration file look like to add a column?

- [ ] **Step 9: Commit**

```bash
git add geoagent/db tests/unit/test_db_connection.py tests/integration/conftest.py tests/integration/test_migrations.py tests/integration/test_workspaces.py
git commit -m "feat: add schema migration, runner, connection retries and workspace helpers"
```

---

## Task 5: Blob stores [LLM zone]

**Files:**
- Create: `geoagent/blobstore/base.py`, `geoagent/blobstore/local.py`, `geoagent/blobstore/gcs.py`
- Test: `tests/unit/test_blobstore.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_blobstore.py`:

```python
from types import SimpleNamespace

from geoagent.blobstore.base import BlobStore
from geoagent.blobstore.gcs import GcsBlobStore
from geoagent.blobstore.local import LocalFsBlobStore


def test_local_put_get_roundtrip(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    uri = store.put("raw/ws/doc.pdf", b"%PDF-1.7 data", "application/pdf")
    assert uri == "local://raw/raw/ws/doc.pdf"
    assert store.get("raw/ws/doc.pdf") == b"%PDF-1.7 data"
    assert (tmp_path / "raw" / "raw" / "ws" / "doc.pdf").exists()


def test_local_list_keys_is_sorted_and_prefix_filtered(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    store.put("incoming/ws/b.pdf", b"b")
    store.put("incoming/ws/a.pdf", b"a")
    store.put("raw/ws/c.pdf", b"c")
    assert store.list_keys("incoming/ws/") == ["incoming/ws/a.pdf", "incoming/ws/b.pdf"]
    assert store.list_keys("nothing/") == []


def test_local_rejects_path_traversal(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    try:
        store.put("../escape.pdf", b"x")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a key escaping the bucket")


def test_local_store_satisfies_protocol(tmp_path):
    assert isinstance(LocalFsBlobStore(root=tmp_path, bucket="raw"), BlobStore)


def test_gcs_uri_and_put_use_bucket():
    uploaded = {}

    class FakeBlob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, data, content_type):
            uploaded[self.name] = (data, content_type)

    fake_bucket = SimpleNamespace(blob=FakeBlob)
    fake_client = SimpleNamespace(bucket=lambda name: fake_bucket)
    store = GcsBlobStore(client=fake_client, bucket="my-bucket")
    assert store.put("raw/ws/x.pdf", b"x", "application/pdf") == "gs://my-bucket/raw/ws/x.pdf"
    assert uploaded["raw/ws/x.pdf"] == (b"x", "application/pdf")
    assert isinstance(store, BlobStore)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_blobstore.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.blobstore.base'`.

- [ ] **Step 3: Implement `geoagent/blobstore/base.py`**

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Opaque object storage. Keys look like 'raw/<workspace>/<document_id>.pdf'."""

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes under key and return the object's URI."""
        ...

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> list[str]:
        """Return keys starting with prefix, sorted."""
        ...

    def uri_for(self, key: str) -> str: ...
```

- [ ] **Step 4: Implement `geoagent/blobstore/local.py`**

```python
from pathlib import Path


class LocalFsBlobStore:
    """Filesystem stand-in for a cloud bucket: <root>/<bucket>/<key>."""

    def __init__(self, root: Path, bucket: str) -> None:
        self.bucket = bucket
        self.base = (Path(root) / bucket).resolve()

    def _path(self, key: str) -> Path:
        path = (self.base / key).resolve()
        if not path.is_relative_to(self.base):
            raise ValueError(f"key escapes the bucket: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        if not self.base.exists():
            return []
        keys = (p.relative_to(self.base).as_posix() for p in self.base.rglob("*") if p.is_file())
        return sorted(k for k in keys if k.startswith(prefix))

    def uri_for(self, key: str) -> str:
        return f"local://{self.bucket}/{key}"
```

- [ ] **Step 5: Implement `geoagent/blobstore/gcs.py`**

```python
from typing import Any


class GcsBlobStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self.bucket_name = bucket
        self._bucket = client.bucket(bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._bucket.blob(key).upload_from_string(data, content_type=content_type)
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(blob.name for blob in self._bucket.list_blobs(prefix=prefix))

    def uri_for(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{key}"
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/unit/test_blobstore.py -q`
Expected: `5 passed`.

- [ ] **Step 7: Commit**

```bash
git add geoagent/blobstore tests/unit/test_blobstore.py
git commit -m "feat: add BlobStore protocol with local filesystem and GCS implementations"
```

---

## Task 6: Provider contracts [Developer zone]

**Files:**
- Create (DEV): `geoagent/providers/types.py`, `geoagent/providers/errors.py`
- Create (LLM): `tests/fakes.py`, `tests/unit/test_provider_contracts.py`

**Contract (type it in yourself — these declarations are the interface every other task depends on):**

`geoagent/providers/types.py`
- `class TaskType(StrEnum)` with members `RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"` and `RETRIEVAL_QUERY = "RETRIEVAL_QUERY"`.
- `@dataclass(frozen=True) class Generation` with fields `text: str`, `model: str`, `tokens_in: int`, `tokens_out: int`.
- `@runtime_checkable class LLMProvider(Protocol)` with attribute `model: str` and method `generate(self, system: str, prompt: str) -> Generation`.
- `@runtime_checkable class EmbeddingProvider(Protocol)` with attribute `dim: int` and method `embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]` — returns one **L2-normalised** vector per input text, in input order.

`geoagent/providers/errors.py`
- `class ProviderError(Exception)` — any provider failure.
- `class ProviderUnavailable(ProviderError)` — host unreachable, 5xx, rate-limited after retries.
- `class ProviderTimeout(ProviderError)` — the call exceeded its timeout.

- [ ] **Step 1 [LLM]: Create `tests/fakes.py`**

```python
import hashlib
import math
import random

from geoagent.providers.types import Generation, TaskType

DIM = 768


def unit_vector(seed: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-random unit vector derived from a string."""
    rnd = random.Random(hashlib.sha256(seed.encode()).digest())
    v = [rnd.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def basis_vector(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def normalized(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class FakeEmbedder:
    def __init__(
        self,
        dim: int = DIM,
        fixed: dict[str, list[float]] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.dim = dim
        self.fixed = fixed or {}
        self.fail_with = fail_with
        self.calls: list[tuple[list[str], TaskType]] = []

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        self.calls.append((list(texts), task_type))
        if self.fail_with is not None:
            raise self.fail_with
        return [self.fixed.get(t) or unit_vector(t, self.dim) for t in texts]


class FakeLLM:
    def __init__(
        self, text: str = "answer", model: str = "fake-model", fail_with: Exception | None = None
    ) -> None:
        self.model = model
        self.text = text
        self.fail_with = fail_with
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> Generation:
        self.calls.append((system, prompt))
        if self.fail_with is not None:
            raise self.fail_with
        return Generation(text=self.text, model=self.model, tokens_in=10, tokens_out=5)
```

- [ ] **Step 2 [LLM]: Write the failing contract tests**

`tests/unit/test_provider_contracts.py`:

```python
import dataclasses

import pytest

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import EmbeddingProvider, Generation, LLMProvider, TaskType
from tests.fakes import FakeEmbedder, FakeLLM


def test_task_types_are_gemini_strings():
    assert TaskType.RETRIEVAL_DOCUMENT == "RETRIEVAL_DOCUMENT"
    assert TaskType.RETRIEVAL_QUERY == "RETRIEVAL_QUERY"


def test_generation_is_frozen_dataclass():
    g = Generation(text="t", model="m", tokens_in=1, tokens_out=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.text = "changed"  # type: ignore[misc]


def test_fakes_satisfy_protocols():
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmbedder(), EmbeddingProvider)


def test_object_without_generate_is_not_an_llm_provider():
    class NotAProvider:
        model = "m"

    assert not isinstance(NotAProvider(), LLMProvider)


def test_error_hierarchy():
    assert issubclass(ProviderUnavailable, ProviderError)
    assert issubclass(ProviderTimeout, ProviderError)
    assert not issubclass(ProviderTimeout, ProviderUnavailable)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_provider_contracts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.providers.errors'` (or `.types`).

- [ ] **Step 4 [DEVELOPER]: Write `types.py` and `errors.py` from the contract**

Read first: Python docs on `typing.Protocol` and `@runtime_checkable` (note: `isinstance` only checks that members *exist*, not their signatures). Why a `Protocol` instead of an abstract base class here?

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/unit/test_provider_contracts.py -q`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add geoagent/providers/types.py geoagent/providers/errors.py tests/fakes.py tests/unit/test_provider_contracts.py
git commit -m "feat: add provider protocols and error hierarchy"
```

---

## Task 7: OllamaProvider [Developer zone]

**Files:**
- Create (DEV): `geoagent/providers/ollama.py`
- Test (LLM): `tests/unit/test_ollama_provider.py`

**Contract:**
- `class OllamaProvider` with `__init__(self, base_url: str, model: str, timeout_s: float = 60.0, client: httpx.Client | None = None)`; exposes `self.model`.
- `generate(system, prompt) -> Generation`: `POST {base_url}/api/chat` (tolerate a trailing slash on `base_url`) with JSON body `{"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "stream": false, "think": false, "options": {"temperature": 0}}`.
- Response mapping: `text = body["message"]["content"]`, `model = body["model"]`, `tokens_in = body.get("prompt_eval_count", 0)`, `tokens_out = body.get("eval_count", 0)`.
- Errors: `httpx.TimeoutException` → `ProviderTimeout`; other `httpx.TransportError` → `ProviderUnavailable`; HTTP 5xx → `ProviderUnavailable`; any other non-2xx → `ProviderError` whose message includes the response's `error` text.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/unit/test_ollama_provider.py`:

```python
import json

import httpx
import pytest

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.ollama import OllamaProvider
from geoagent.providers.types import Generation, LLMProvider


def make(handler, base_url: str = "http://gpu-host:11434") -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(base_url=base_url, model="qwen3:14b", timeout_s=5, client=client)


def ok_body(content: str = "Gold [1].") -> dict:
    return {
        "model": "qwen3:14b",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 120,
        "eval_count": 7,
        "done": True,
    }


def test_sends_chat_request_and_parses_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    gen = make(handler).generate(system="SYS", prompt="USER")

    assert seen["method"] == "POST"
    assert seen["url"] == "http://gpu-host:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "qwen3:14b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert gen == Generation(text="Gold [1].", model="qwen3:14b", tokens_in=120, tokens_out=7)


def test_trailing_slash_in_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=ok_body())

    make(handler, base_url="http://gpu-host:11434/").generate("s", "p")
    assert urls == ["http://gpu-host:11434/api/chat"]


def test_missing_token_counts_default_to_zero():
    body = ok_body()
    del body["prompt_eval_count"], body["eval_count"]
    gen = make(lambda r: httpx.Response(200, json=body)).generate("s", "p")
    assert (gen.tokens_in, gen.tokens_out) == (0, 0)


def test_satisfies_protocol():
    assert isinstance(make(lambda r: httpx.Response(200, json=ok_body())), LLMProvider)


def test_timeout_maps_to_provider_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderTimeout):
        make(handler).generate("s", "p")


def test_connection_refused_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailable):
        make(handler).generate("s", "p")


def test_server_error_maps_to_unavailable():
    with pytest.raises(ProviderUnavailable):
        make(lambda r: httpx.Response(500, json={"error": "boom"})).generate("s", "p")


def test_model_not_pulled_is_a_provider_error_with_message():
    resp = httpx.Response(404, json={"error": "model 'qwen3:14b' not found"})
    with pytest.raises(ProviderError) as info:
        make(lambda r: resp).generate("s", "p")
    assert not isinstance(info.value, ProviderUnavailable)
    assert "not found" in str(info.value)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_ollama_provider.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.providers.ollama'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/providers/ollama.py`**

Read first: Ollama API docs for `/api/chat` (https://docs.ollama.com/api), httpx exception hierarchy (https://www.python-httpx.org/exceptions/). Questions to answer in your head: why `"think": false` for reasoning models like Qwen3? Why should a 404 *not* be `ProviderUnavailable`? Why is `temperature: 0` reasonable for grounded QA?

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_ollama_provider.py -q`
Expected: `8 passed`.

- [ ] **Step 5 [LLM]: Review** — check exception chaining (`raise ... from exc`), that the per-request timeout is applied, and that the client is reused (not created per call).

- [ ] **Step 6: Commit**

```bash
git add geoagent/providers/ollama.py tests/unit/test_ollama_provider.py
git commit -m "feat: add Ollama LLM provider"
```

---

## Task 8: Gemini embeddings + Vertex Gemini provider [LLM zone]

**Files:**
- Create: `geoagent/providers/gemini.py`
- Test: `tests/unit/test_gemini_providers.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_gemini_providers.py`:

```python
import math
from types import SimpleNamespace

import httpx
import pytest

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.gemini import GeminiEmbeddings, VertexGeminiProvider, l2_normalize
from geoagent.providers.types import EmbeddingProvider, Generation, LLMProvider, TaskType


class FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.embed_calls: list = []
        self.generate_calls: list = []

    def _next(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def embed_content(self, *, model, contents, config):
        self.embed_calls.append((model, list(contents), config))
        return self._next()

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append((model, contents, config))
        return self._next()


def emb(*vectors) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=list(v)) for v in vectors])


def embedder(models: FakeModels, dim: int = 2, batch_size: int = 2, sleeps: list | None = None):
    record = sleeps if sleeps is not None else []
    return GeminiEmbeddings(
        client=SimpleNamespace(models=models),
        model="gemini-embedding-001",
        dim=dim,
        batch_size=batch_size,
        max_retries=2,
        sleep=record.append,
    )


def test_l2_normalize():
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ProviderError):
        l2_normalize([0.0, 0.0])


def test_embed_normalizes_and_passes_task_type_and_dim():
    models = FakeModels([emb([3, 4])])
    out = embedder(models).embed(["hello"], TaskType.RETRIEVAL_QUERY)
    assert out == [[0.6, 0.8]]
    model, contents, config = models.embed_calls[0]
    assert model == "gemini-embedding-001"
    assert contents == ["hello"]
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == 2


def test_embed_batches_and_preserves_order():
    models = FakeModels([emb([1, 0], [0, 1]), emb([1, 1], [2, 0]), emb([0, 3])])
    out = embedder(models, batch_size=2).embed(list("abcde"), TaskType.RETRIEVAL_DOCUMENT)
    assert [len(call[1]) for call in models.embed_calls] == [2, 2, 1]
    assert len(out) == 5
    assert out[2] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert out[4] == [0.0, 1.0]


def test_empty_input_makes_no_calls():
    models = FakeModels([])
    assert embedder(models).embed([], TaskType.RETRIEVAL_DOCUMENT) == []
    assert models.embed_calls == []


def test_retries_rate_limit_then_succeeds():
    sleeps: list = []
    models = FakeModels([FakeAPIError(429), emb([0, 2])])
    out = embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)
    assert out == [[0.0, 1.0]]
    assert sleeps == [1]


def test_gives_up_after_max_retries():
    models = FakeModels([FakeAPIError(503), FakeAPIError(503), FakeAPIError(503)])
    with pytest.raises(ProviderUnavailable):
        embedder(models).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_client_error_is_not_retried():
    sleeps: list = []
    models = FakeModels([FakeAPIError(400)])
    with pytest.raises(ProviderError) as info:
        embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)
    assert not isinstance(info.value, ProviderUnavailable)
    assert sleeps == []


def test_wrong_dimension_is_rejected():
    models = FakeModels([emb([1, 2, 3])])
    with pytest.raises(ProviderError):
        embedder(models, dim=2).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_embedder_satisfies_protocol():
    assert isinstance(embedder(FakeModels([])), EmbeddingProvider)


def vertex(models: FakeModels) -> VertexGeminiProvider:
    return VertexGeminiProvider(client=SimpleNamespace(models=models), model="gemini-3.8-flash")


def test_vertex_generate_maps_text_and_usage():
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=4)
    models = FakeModels([SimpleNamespace(text="Gold [1].", usage_metadata=usage)])
    gen = vertex(models).generate(system="SYS", prompt="USER")
    assert gen == Generation(text="Gold [1].", model="gemini-3.8-flash", tokens_in=10, tokens_out=7)
    model, contents, config = models.generate_calls[0]
    assert model == "gemini-3.8-flash"
    assert contents == "USER"
    assert config.system_instruction == "SYS"


def test_vertex_missing_usage_defaults_to_zero():
    models = FakeModels([SimpleNamespace(text=None, usage_metadata=None)])
    gen = vertex(models).generate("s", "p")
    assert (gen.text, gen.tokens_in, gen.tokens_out) == ("", 0, 0)


def test_vertex_error_mapping():
    req = httpx.Request("POST", "https://example.invalid")
    with pytest.raises(ProviderTimeout):
        vertex(FakeModels([httpx.ReadTimeout("slow", request=req)])).generate("s", "p")
    with pytest.raises(ProviderUnavailable):
        vertex(FakeModels([FakeAPIError(503)])).generate("s", "p")
    with pytest.raises(ProviderError):
        vertex(FakeModels([FakeAPIError(400)])).generate("s", "p")


def test_vertex_satisfies_protocol():
    assert isinstance(vertex(FakeModels([])), LLMProvider)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_gemini_providers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.providers.gemini'`.

- [ ] **Step 3: Implement `geoagent/providers/gemini.py`**

```python
import math
import time
from collections.abc import Callable
from typing import Any

import httpx
from google.genai import types

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import Generation, TaskType

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ProviderError("embedding has zero norm")
    return [x / norm for x in vector]


class GeminiEmbeddings:
    """gemini-embedding-001 via google-genai (AI Studio key locally, Vertex AI in cloud).

    Vectors smaller than 3072 dims are NOT normalised by the API, so we normalise here.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        dim: int,
        batch_size: int = 32,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sleep = sleep

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size], task_type))
        return vectors

    def _embed_batch(self, batch: list[str], task_type: TaskType) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type.value, output_dimensionality=self.dim
        )
        for attempt in range(self.max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(f"embedding request timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise ProviderUnavailable(f"embedding service unreachable: {exc}") from exc
            except Exception as exc:
                code = _status_code(exc)
                if code in RETRYABLE_CODES and attempt < self.max_retries:
                    self.sleep(min(2**attempt, 30))
                    continue
                if code in RETRYABLE_CODES:
                    raise ProviderUnavailable(f"embedding failed after retries: {exc}") from exc
                raise ProviderError(f"embedding failed: {exc}") from exc
            values = [list(e.values) for e in result.embeddings]
            if len(values) != len(batch):
                raise ProviderError(f"expected {len(batch)} embeddings, got {len(values)}")
            for v in values:
                if len(v) != self.dim:
                    raise ProviderError(f"expected dim {self.dim}, got {len(v)}")
            return [l2_normalize(v) for v in values]
        raise AssertionError("unreachable")


class VertexGeminiProvider:
    """Gemini on Vertex AI. Temperature is left at the model default (Gemini 3 guidance)."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, system: str, prompt: str) -> Generation:
        config = types.GenerateContentConfig(system_instruction=system)
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Gemini request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"Gemini unreachable: {exc}") from exc
        except Exception as exc:
            if _status_code(exc) in RETRYABLE_CODES:
                raise ProviderUnavailable(f"Gemini unavailable: {exc}") from exc
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        usage = resp.usage_metadata
        tokens_in = (getattr(usage, "prompt_token_count", None) or 0) if usage else 0
        tokens_out = (
            (getattr(usage, "candidates_token_count", None) or 0)
            + (getattr(usage, "thoughts_token_count", None) or 0)
            if usage
            else 0
        )
        return Generation(
            text=resp.text or "", model=self.model, tokens_in=tokens_in, tokens_out=tokens_out
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_gemini_providers.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Developer review checklist** — Why is `batch_size` 32 (Vertex: ≤250 texts **and** ≤20k tokens per request; 32 × 512 = 16,384)? Why do thinking tokens count as output tokens (billing)? Why normalise here rather than in SQL?

- [ ] **Step 6: Commit**

```bash
git add geoagent/providers/gemini.py tests/unit/test_gemini_providers.py
git commit -m "feat: add Gemini embeddings and Vertex Gemini provider"
```

---

## Task 9: Synthetic fixture PDF [LLM zone]

A small, fictional NI 43-101-style PDF generated in code (nothing binary is committed). It has a repeated header, a numbered footer, numbered section headings, a section crossing a page break, and a ruled table.

**Files:**
- Create: `tests/fixtures/fixture_pdf.py`
- Test: `tests/unit/test_fixture_pdf.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fixture_pdf.py`:

```python
import pymupdf

from tests.fixtures.fixture_pdf import HEADER, build_fixture_pdf


def test_fixture_has_four_pages_with_header_and_footer():
    doc = pymupdf.open(stream=build_fixture_pdf(), filetype="pdf")
    assert doc.page_count == 4
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        assert HEADER in text
        assert f"Page {i} of 4" in text


def test_variants_differ():
    assert build_fixture_pdf("a") != build_fixture_pdf("b")


def test_page_three_has_one_detectable_table():
    doc = pymupdf.open(stream=build_fixture_pdf(), filetype="pdf")
    tables = doc[2].find_tables().tables
    assert len(tables) == 1
    assert tables[0].extract()[2][0].strip() == "DH-002"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_fixture_pdf.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures.fixture_pdf'`.

- [ ] **Step 3: Implement `tests/fixtures/fixture_pdf.py`**

```python
"""Fictional NI 43-101-style report used by parser/chunker/pipeline tests."""

import pymupdf

HEADER = "Kiwi Ridge Project - NI 43-101 Technical Report"

TABLE_ROWS = [
    ["Hole", "From (m)", "To (m)", "Au (g/t)"],
    ["DH-001", "45.0", "52.0", "3.10"],
    ["DH-002", "112.0", "118.5", "4.70"],
]


def _pages(variant: str) -> list[list[str]]:
    tonnes = "1.2" if variant == "a" else "1.5"
    return [
        [
            "1 SUMMARY",
            "The Kiwi Ridge Project is a fictional epithermal gold prospect.",
            "This report exists only to exercise automated tests.",
            "2 INTRODUCTION",
            "The report follows the structure of an NI 43-101 technical report.",
            "Drilling and sampling were completed between 2021 and 2023.",
        ],
        [
            "7 GEOLOGICAL SETTING AND MINERALIZATION",
            "7.1 Regional Geology",
            "The project area is underlain by greywacke basement rocks.",
            "The greywacke is overlain by andesitic volcanic flows and tuffs.",
            "Regional faults trend north-east across the greywacke terrane.",
        ],
        [
            "Weathering of the greywacke extends to a depth of thirty metres.",
            "7.2 Mineralization",
            "Gold occurs in quartz-sulphide veins hosted by the andesite.",
            "The quartz-sulphide veins dip steeply to the north-west.",
            "Significant intercepts are listed in the table below.",
        ],
        [
            "14 MINERAL RESOURCE ESTIMATES",
            f"The Indicated Mineral Resource is {tonnes} Mt at 2.3 g/t Au.",
            "The Inferred Mineral Resource is 0.4 Mt at 1.8 g/t Au.",
        ],
    ]


def _draw_table(page: pymupdf.Page, x: float, y: float) -> None:
    col_w, row_h = 110.0, 18.0
    for r, row in enumerate(TABLE_ROWS):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x + c * col_w, y + r * row_h, x + (c + 1) * col_w, y + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 3, rect.y1 - 5), cell, fontsize=9)


def build_fixture_pdf(variant: str = "a") -> bytes:
    pages = _pages(variant)
    doc = pymupdf.open()
    for number, lines in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), HEADER, fontsize=9)
        y = 80.0
        for line in lines:
            page.insert_text((50, y), line, fontsize=10)
            y += 16
        if number == 3:
            _draw_table(page, 50, y + 10)
        page.insert_text((260, 815), f"Page {number} of {len(pages)}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_fixture_pdf.py -q`
Expected: `3 passed`. If the table test fails, PyMuPDF did not recognise the drawn grid: adjust `_draw_table` (e.g. thicker lines, `page.draw_line` per border) until it passes — the parser task depends on it.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/fixture_pdf.py tests/unit/test_fixture_pdf.py
git commit -m "test: add synthetic NI 43-101-style fixture PDF generator"
```

---

## Task 10: PDF parser [Developer zone]

**Files:**
- Create (DEV): `geoagent/ingest/parse.py`
- Test (LLM): `tests/unit/test_parse.py`

**Contract:**
- `@dataclass(frozen=True) class Page` with `number: int` (1-based) and `text: str`.
- `def parse_pdf(data: bytes) -> list[Page]` — one `Page` per PDF page, in order. `text` is the page's plain text (PyMuPDF `page.get_text()`), followed by any tables PyMuPDF detects (`page.find_tables()`), each table row rendered on its own line as cells joined by `" | "` (cells stripped; `None` cells become empty strings).
- Raises `ValueError` if `data` is not a readable PDF.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/unit/test_parse.py`:

```python
import pytest

from geoagent.ingest.parse import Page, parse_pdf
from tests.fixtures.fixture_pdf import HEADER, build_fixture_pdf


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    return parse_pdf(build_fixture_pdf())


def test_one_page_object_per_pdf_page_numbered_from_one(pages):
    assert [p.number for p in pages] == [1, 2, 3, 4]


def test_page_text_contains_body_header_and_footer(pages):
    assert "1 SUMMARY" in pages[0].text
    assert HEADER in pages[1].text
    assert "Page 3 of 4" in pages[2].text


def test_table_rows_rendered_pipe_delimited(pages):
    assert "Hole | From (m) | To (m) | Au (g/t)" in pages[2].text
    assert "DH-002 | 112.0 | 118.5 | 4.70" in pages[2].text


def test_page_is_frozen(pages):
    with pytest.raises(AttributeError):
        pages[0].text = "x"  # type: ignore[misc]


def test_not_a_pdf_raises_value_error():
    with pytest.raises(ValueError):
        parse_pdf(b"definitely not a pdf")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_parse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.ingest.parse'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/ingest/parse.py`**

Read first: PyMuPDF "Text extraction" and `Page.find_tables()` docs (https://pymupdf.readthedocs.io). Things to notice: what exception does `pymupdf.open(stream=..., filetype="pdf")` raise on garbage bytes? Table cells also appear in `get_text()` output — is that duplication a problem for embeddings? (Note it for the learning log; no need to fix now.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_parse.py -q`
Expected: `5 passed`.

- [ ] **Step 5 [DEVELOPER]: Look at real output** — after Task 17 downloads the reports, run `uv run python -c "from geoagent.ingest.parse import parse_pdf; p=parse_pdf(open('data/macraes-ni43-101.pdf','rb').read()); print(len(p)); print(p[50].text)"` and inspect a few pages. Write what surprised you in the learning log.

- [ ] **Step 6: Commit**

```bash
git add geoagent/ingest/parse.py tests/unit/test_parse.py
git commit -m "feat: add PDF parser with table rendering"
```

---

## Task 11: Chunker [Developer zone]

**Files:**
- Create (DEV): `geoagent/ingest/chunker.py`
- Test (LLM): `tests/unit/test_chunker.py`

**Contract:**
- `def count_tokens(text: str) -> int` — tokens per `tiktoken` encoding `cl100k_base` (a proxy for Gemini's tokenizer; keep a safety margin below the 2,048-token embedding input limit).
- `@dataclass(frozen=True) class Chunk` with `ordinal: int`, `page_start: int`, `page_end: int`, `section: str | None`, `text: str`, `token_count: int`.
- `def chunk_pages(pages: list[Page], max_tokens: int = 512, overlap_tokens: int = 64, strip_headers: bool = True) -> list[Chunk]` with these rules:
  1. **Header/footer stripping** (when `strip_headers`): a line is boilerplate if, after replacing every digit with `#` and stripping whitespace, it occurs on at least half of the pages (and the document has ≥ 2 pages). Remove those lines everywhere.
  2. **Sections:** a line matching a numbered heading (e.g. `7 GEOLOGICAL SETTING AND MINERALIZATION`, `7.1 Regional Geology`, `14 MINERAL RESOURCE ESTIMATES`) starts a new section; the heading line itself (stripped) becomes `section`. Text before the first heading has `section=None`. A heading with no body text before the next heading produces no chunk.
  3. **Boundaries:** a chunk never contains text from two different sections. Sections may span pages; `page_start`/`page_end` are the first/last pages contributing text to the chunk.
  4. **Size:** every chunk has `token_count == count_tokens(text) <= max_tokens`. A section longer than `max_tokens` is split into several chunks; consecutive chunks of the same section overlap by roughly `overlap_tokens` tokens (the next chunk starts with words from the end of the previous one).
  5. `ordinal` runs 0..n-1 in document order. No chunk has empty/whitespace-only text.
  6. `overlap_tokens >= max_tokens` raises `ValueError`. `pages == []` returns `[]`.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/unit/test_chunker.py`:

```python
import re

import pytest

from geoagent.ingest.chunker import Chunk, chunk_pages, count_tokens
from geoagent.ingest.parse import Page, parse_pdf
from tests.fixtures.fixture_pdf import HEADER, build_fixture_pdf


@pytest.fixture(scope="module")
def fixture_chunks() -> list[Chunk]:
    return chunk_pages(parse_pdf(build_fixture_pdf()))


def test_count_tokens_is_positive_and_monotonic():
    assert count_tokens("") == 0
    assert 0 < count_tokens("gold") < count_tokens("gold in quartz-sulphide veins")


def test_sections_detected(fixture_chunks):
    assert {c.section for c in fixture_chunks} == {
        "1 SUMMARY",
        "2 INTRODUCTION",
        "7.1 Regional Geology",
        "7.2 Mineralization",
        "14 MINERAL RESOURCE ESTIMATES",
    }


def test_ordinals_contiguous_and_text_non_empty(fixture_chunks):
    assert [c.ordinal for c in fixture_chunks] == list(range(len(fixture_chunks)))
    assert all(c.text.strip() for c in fixture_chunks)


def test_token_counts_are_exact_and_bounded(fixture_chunks):
    for c in fixture_chunks:
        assert c.token_count == count_tokens(c.text)
        assert c.token_count <= 512


def test_no_chunk_mixes_sections(fixture_chunks):
    for c in fixture_chunks:
        assert not ("greywacke" in c.text and "quartz-sulphide" in c.text)


def test_section_crossing_a_page_break_tracks_both_pages(fixture_chunks):
    regional = [c for c in fixture_chunks if c.section == "7.1 Regional Geology"]
    assert regional[0].page_start == 2
    assert regional[-1].page_end == 3
    assert any("Weathering of the greywacke" in c.text for c in regional)


def test_page_ranges_are_valid(fixture_chunks):
    for c in fixture_chunks:
        assert 1 <= c.page_start <= c.page_end <= 4


def test_headers_and_footers_are_stripped(fixture_chunks):
    for c in fixture_chunks:
        assert HEADER not in c.text
        assert not re.search(r"Page \d+ of \d+", c.text)


def test_headers_kept_when_stripping_disabled():
    chunks = chunk_pages(parse_pdf(build_fixture_pdf()), strip_headers=False)
    assert any(HEADER in c.text for c in chunks)


def test_long_section_split_with_overlap():
    words = " ".join(f"w{i}" for i in range(3000))
    pages = [Page(number=1, text="9 DRILLING\n" + words)]
    chunks = chunk_pages(pages, max_tokens=200, overlap_tokens=20)
    assert len(chunks) > 5
    assert all(c.section == "9 DRILLING" for c in chunks)
    assert all(c.token_count <= 200 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text.split()[0] in prev.text.split(), "next chunk should start inside previous"
    covered = set(" ".join(c.text for c in chunks).split())
    assert {f"w{i}" for i in range(3000)} <= covered


def test_text_before_first_heading_has_no_section():
    chunks = chunk_pages([Page(number=1, text="Preface text.\n1 SUMMARY\nBody text.")])
    assert chunks[0].section is None
    assert chunks[1].section == "1 SUMMARY"


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_pages([Page(number=1, text="x")], max_tokens=100, overlap_tokens=100)


def test_empty_input():
    assert chunk_pages([]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_chunker.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.ingest.chunker'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/ingest/chunker.py`**

This is the most important module of the sprint — take your time. Suggested order: (1) `count_tokens`; (2) boilerplate-line detection; (3) turn pages into a stream of `(page_number, line)` with headers removed; (4) group lines into sections with a heading regex; (5) pack each section's lines into chunks under the token limit, splitting over-long lines by words; (6) add overlap; (7) assign ordinals. Write the heading regex yourself and test it on lines from the real reports — what lines would wrongly match `^\d+(\.\d+)*\s+\S`? (e.g. `10 m`, `2023 drilling`). Hint: `tiktoken.get_encoding` is slow — create the encoding once at module level.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_chunker.py -q`
Expected: `13 passed`.

- [ ] **Step 5 [LLM]: Review** — ask: what happens with a 3,000-token table with no newlines? With a heading regex that matches table rows like `1 | 45.0 | ...`? How would section-title prefixing (`"7.1 Regional Geology: <text>"`) change embeddings?

- [ ] **Step 6: Commit**

```bash
git add geoagent/ingest/chunker.py tests/unit/test_chunker.py
git commit -m "feat: add section-aware, token-limited chunker"
```

---

## Task 12: Ingestion pipeline [Developer zone]

> **Amendment — schema v2 (apply before executing this task):** `workspace_id` is a `uuid.UUID`, not a string. Contract step 1 ("ensure the workspace row exists") is **removed**: callers resolve or create the workspace first (`geoagent.db.workspaces`). In the tests, replace the `WS` constant with a fixture `ws` returning `create_workspace(conn, slug="test-ws", name="Test")`, and delete the assertion on `SELECT id FROM workspaces`. Blob keys use the UUID: `raw/{workspace_id}/{document_id}.pdf`, uploads under `incoming/{workspace_id}/`. **Review note (Task 4):** upsert the document with `INSERT ... ON CONFLICT (workspace_id, sha256) DO UPDATE ... RETURNING id` (also resolves two runners ingesting the same PDF); never UPDATE `documents.id`/`workspace_id` (composite FK target); write the `failed` status on the autocommit connection outside the chunk-insert transaction; set `updated_at = now()` explicitly (no trigger).

**Files:**
- Create (DEV): `geoagent/ingest/pipeline.py`
- Test (LLM): `tests/integration/test_pipeline.py`

**Contract:**
- `@dataclass(frozen=True) class IngestResult` with `document_id: UUID`, `title: str`, `status: str` (`"ready"` or `"failed"`), `skipped: bool`, `error: str | None = None`.
- `def ingest_bytes(conn, blobs: BlobStore, embedder: EmbeddingProvider, *, workspace_id: str, data: bytes, title: str, source_url: str | None = None, max_tokens: int = 512, overlap_tokens: int = 64) -> IngestResult`
  1. Ensure the workspace row exists (`name` = `workspace_id`).
  2. `sha256` of `data`. If a document with `(workspace_id, sha256)` exists with status `ready` → return it with `skipped=True` (no blob write, no embedding call).
  3. Otherwise reuse the existing document id (retry after failure) or create a new `uuid4`; write the blob to key `raw/{workspace_id}/{document_id}.pdf` (`content_type="application/pdf"`); upsert the `documents` row with `blob_uri`, `title`, `source_url`, status `uploaded`.
  4. Set status `processing`; `parse_pdf` → `chunk_pages(max_tokens, overlap_tokens)` → `embedder.embed([c.text ...], TaskType.RETRIEVAL_DOCUMENT)` (one call with all texts — the embedder batches internally).
  5. In **one transaction**: delete existing chunks of the document, insert new chunks (`uuid4` ids, embeddings as `numpy.array`), set `status='ready'`, `page_count`, `error=NULL`, `updated_at=now()`.
  6. On any exception in 3–5: set status `failed` and `error=str(exc)`, return `IngestResult(status="failed", error=...)`. Never raise.
- `def ingest_from_blob(conn, blobs, embedder, *, workspace_id: str, prefix: str) -> list[IngestResult]` — for each key from `blobs.list_keys(prefix)` ending in `.pdf` (case-insensitive): `ingest_bytes(..., data=blobs.get(key), title=PurePosixPath(key).stem, source_url=blobs.uri_for(key))`.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/integration/test_pipeline.py`:

```python
import hashlib

import pytest

from geoagent.blobstore.local import LocalFsBlobStore
from geoagent.ingest.chunker import chunk_pages
from geoagent.ingest.parse import parse_pdf
from geoagent.ingest.pipeline import ingest_bytes, ingest_from_blob
from geoagent.providers.errors import ProviderUnavailable
from geoagent.providers.types import TaskType
from tests.fakes import FakeEmbedder
from tests.fixtures.fixture_pdf import build_fixture_pdf

pytestmark = pytest.mark.integration

WS = "test-ws"


@pytest.fixture
def blobs(tmp_path) -> LocalFsBlobStore:
    return LocalFsBlobStore(root=tmp_path, bucket="raw")


def doc_row(conn, document_id):
    return conn.execute(
        "SELECT status, error, page_count, sha256, blob_uri, title FROM documents WHERE id = %s",
        (document_id,),
    ).fetchone()


def chunk_count(conn, document_id) -> int:
    return conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,)
    ).fetchone()[0]


def test_ingest_stores_blob_document_and_chunks(conn, blobs):
    data = build_fixture_pdf()
    embedder = FakeEmbedder()
    result = ingest_bytes(conn, blobs, embedder, workspace_id=WS, data=data, title="Kiwi Ridge")

    assert result.status == "ready" and result.skipped is False and result.error is None
    status, error, page_count, sha, blob_uri, title = doc_row(conn, result.document_id)
    assert (status, error, page_count, title) == ("ready", None, 4, "Kiwi Ridge")
    assert sha == hashlib.sha256(data).hexdigest()
    key = f"raw/{WS}/{result.document_id}.pdf"
    assert blob_uri == blobs.uri_for(key)
    assert blobs.get(key) == data
    assert chunk_count(conn, result.document_id) == len(chunk_pages(parse_pdf(data)))
    assert conn.execute("SELECT id FROM workspaces").fetchall() == [(WS,)]


def test_chunks_are_embedded_as_documents(conn, blobs):
    embedder = FakeEmbedder()
    ingest_bytes(conn, blobs, embedder, workspace_id=WS, data=build_fixture_pdf(), title="t")
    assert embedder.calls and all(tt == TaskType.RETRIEVAL_DOCUMENT for _, tt in embedder.calls)


def test_chunk_rows_carry_workspace_pages_and_section(conn, blobs):
    r = ingest_bytes(conn, blobs, FakeEmbedder(), workspace_id=WS, data=build_fixture_pdf(), title="t")
    rows = conn.execute(
        "SELECT workspace_id, page_start, page_end, section, token_count FROM chunks "
        "WHERE document_id = %s ORDER BY ordinal",
        (r.document_id,),
    ).fetchall()
    assert all(row[0] == WS for row in rows)
    assert ("7.1 Regional Geology" in {row[3] for row in rows})
    assert all(row[1] <= row[2] and row[4] > 0 for row in rows)


def test_reingesting_same_bytes_is_skipped(conn, blobs):
    data = build_fixture_pdf()
    first = ingest_bytes(conn, blobs, FakeEmbedder(), workspace_id=WS, data=data, title="t")
    embedder = FakeEmbedder()
    second = ingest_bytes(conn, blobs, embedder, workspace_id=WS, data=data, title="t")
    assert second.skipped is True and second.status == "ready"
    assert second.document_id == first.document_id
    assert embedder.calls == []
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (1,)


def test_embedding_failure_marks_document_failed(conn, blobs):
    failing = FakeEmbedder(fail_with=ProviderUnavailable("quota exhausted"))
    r = ingest_bytes(conn, blobs, failing, workspace_id=WS, data=build_fixture_pdf(), title="t")
    assert r.status == "failed" and "quota exhausted" in (r.error or "")
    status, error, *_ = doc_row(conn, r.document_id)
    assert status == "failed" and "quota exhausted" in error
    assert chunk_count(conn, r.document_id) == 0


def test_retry_after_failure_reuses_document_and_succeeds(conn, blobs):
    data = build_fixture_pdf()
    failed = ingest_bytes(
        conn, blobs, FakeEmbedder(fail_with=ProviderUnavailable("down")),
        workspace_id=WS, data=data, title="t",
    )
    ok = ingest_bytes(conn, blobs, FakeEmbedder(), workspace_id=WS, data=data, title="t")
    assert ok.status == "ready" and ok.skipped is False
    assert ok.document_id == failed.document_id
    assert chunk_count(conn, ok.document_id) == len(chunk_pages(parse_pdf(data)))


def test_invalid_pdf_fails_without_raising(conn, blobs):
    r = ingest_bytes(conn, blobs, FakeEmbedder(), workspace_id=WS, data=b"nope", title="bad")
    assert r.status == "failed" and r.error


def test_ingest_from_blob_processes_every_pdf_under_prefix(conn, blobs):
    blobs.put(f"incoming/{WS}/alpha.pdf", build_fixture_pdf("a"))
    blobs.put(f"incoming/{WS}/beta.PDF", build_fixture_pdf("b"))
    blobs.put(f"incoming/{WS}/notes.txt", b"ignore me")
    results = ingest_from_blob(conn, blobs, FakeEmbedder(), workspace_id=WS, prefix=f"incoming/{WS}/")
    assert sorted(r.title for r in results) == ["alpha", "beta"]
    assert all(r.status == "ready" for r in results)
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (2,)
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose up -d postgres` then `uv run pytest tests/integration/test_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.ingest.pipeline'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/ingest/pipeline.py`**

Read first: psycopg 3 "Transactions management" (https://www.psycopg.org/psycopg3/docs/basic/transactions.html) and pgvector-python's psycopg 3 section (https://github.com/pgvector/pgvector-python — pass vectors as `numpy.array`). Think about: which statements must share a transaction, and which must *not* (the `failed` status update must survive a rollback)? Why is the sha check what makes re-running the cloud job safe?

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/integration/test_pipeline.py -q`
Expected: `8 passed`.

- [ ] **Step 5 [LLM]: Review** — check: is there any window where a document is `ready` with zero chunks? What happens if two job executions ingest the same file concurrently (hint: the `UNIQUE` constraint)? Record the answer in the learning log.

- [ ] **Step 6: Commit**

```bash
git add geoagent/ingest/pipeline.py tests/integration/test_pipeline.py
git commit -m "feat: add idempotent ingestion pipeline with status tracking"
```

---

## Task 13: Retriever [Developer zone]

> **Amendment — schema v2 (apply before executing this task):** `workspace_id` is a `uuid.UUID`. `tests/integration/seed.py` gains `ensure_workspace(conn, slug) -> UUID` (get-or-create via `geoagent.db.workspaces`); `seed_chunk` takes `workspace_slug: str`, inserts the document/chunk with the resolved UUID and the new `workspaces(id, slug, name)` columns. The `seeded` fixture also returns `ws_a`/`ws_b` UUIDs; tests call `retrieve(..., workspace_id=seeded["ws_a"])`; the unknown-workspace test uses `uuid.uuid4()`. **Review note (Task 4) — HNSW + tenant filter:** with the default `hnsw.iterative_scan = off`, once a big tenant exists the planner may use HNSW, scan only `ef_search` (40) candidates, then post-filter → a small tenant gets fewer than k rows. Run the query inside `with conn.transaction():` after `SET LOCAL hnsw.iterative_scan = relaxed_order` (autocommit makes `SET LOCAL` outside a transaction a silent no-op); `relaxed_order` is only approximately ordered, so wrap it in an outer `ORDER BY distance`. Add an integration test with one big tenant (thousands of rows) and one small tenant (a handful) asserting the small tenant still gets `top_k` results. `retrieve()` does not look up workspaces (the API resolves the slug once).

**Files:**
- Create (DEV): `geoagent/rag/retriever.py`
- Create (LLM): `tests/integration/seed.py`
- Test (LLM): `tests/integration/test_retriever.py`

**Contract:**
- `@dataclass(frozen=True) class RetrievedChunk` with `chunk_id: UUID`, `document_id: UUID`, `document_title: str`, `page_start: int`, `page_end: int`, `section: str | None`, `text: str`, `similarity: float`.
- `def retrieve(conn, embedder: EmbeddingProvider, *, workspace_id: str, question: str, top_k: int) -> list[RetrievedChunk]` — embed `question` with `TaskType.RETRIEVAL_QUERY`; return at most `top_k` chunks **of that workspace only**, ordered by cosine distance ascending (`<=>`), with `similarity = 1 - distance`.

- [ ] **Step 1 [LLM]: Create `tests/integration/seed.py`**

```python
import hashlib
import uuid

import numpy as np


def seed_chunk(
    conn,
    *,
    workspace_id: str,
    text: str,
    vector: list[float],
    document_title: str = "Doc",
    page: int = 1,
    section: str | None = None,
) -> uuid.UUID:
    """Insert workspace + one ready document + one chunk with an explicit vector."""
    conn.execute(
        "INSERT INTO workspaces (id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (workspace_id, workspace_id),
    )
    doc_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO documents (id, workspace_id, title, blob_uri, sha256, status, page_count) "
        "VALUES (%s, %s, %s, %s, %s, 'ready', 1)",
        (doc_id, workspace_id, document_title, f"local://raw/{doc_id}.pdf",
         hashlib.sha256(doc_id.bytes).hexdigest()),
    )
    chunk_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO chunks (id, document_id, workspace_id, ordinal, page_start, page_end, "
        "section, text, token_count, embedding) VALUES (%s, %s, %s, 0, %s, %s, %s, %s, %s, %s)",
        (chunk_id, doc_id, workspace_id, page, page, section, text, len(text.split()),
         np.array(vector, dtype=np.float32)),
    )
    return chunk_id
```

- [ ] **Step 2 [LLM]: Write the failing tests**

`tests/integration/test_retriever.py`:

```python
import math

import pytest

from geoagent.providers.types import TaskType
from geoagent.rag.retriever import RetrievedChunk, retrieve
from tests.fakes import FakeEmbedder, basis_vector, normalized
from tests.integration.seed import seed_chunk

pytestmark = pytest.mark.integration

Q = "where is the gold?"


@pytest.fixture
def seeded(conn):
    ids = {
        "exact": seed_chunk(conn, workspace_id="ws-a", text="exact", vector=basis_vector(0),
                            document_title="Report A", page=7, section="7.2 Mineralization"),
        "near": seed_chunk(conn, workspace_id="ws-a", text="near",
                           vector=normalized([1.0, 1.0] + [0.0] * 766)),
        "far": seed_chunk(conn, workspace_id="ws-a", text="far", vector=basis_vector(1)),
        "other_ws": seed_chunk(conn, workspace_id="ws-b", text="secret", vector=basis_vector(0)),
    }
    return ids


def embedder() -> FakeEmbedder:
    return FakeEmbedder(fixed={Q: basis_vector(0)})


def test_orders_by_similarity_and_respects_top_k(conn, seeded):
    results = retrieve(conn, embedder(), workspace_id="ws-a", question=Q, top_k=2)
    assert [r.text for r in results] == ["exact", "near"]
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert results[1].similarity == pytest.approx(1 / math.sqrt(2), abs=1e-5)


def test_returns_all_when_top_k_exceeds_rows(conn, seeded):
    results = retrieve(conn, embedder(), workspace_id="ws-a", question=Q, top_k=50)
    assert [r.text for r in results] == ["exact", "near", "far"]
    assert results[2].similarity == pytest.approx(0.0, abs=1e-5)


def test_never_returns_other_workspaces(conn, seeded):
    results = retrieve(conn, embedder(), workspace_id="ws-a", question=Q, top_k=50)
    assert "secret" not in {r.text for r in results}


def test_populates_metadata(conn, seeded):
    top = retrieve(conn, embedder(), workspace_id="ws-a", question=Q, top_k=1)[0]
    assert isinstance(top, RetrievedChunk)
    assert top.chunk_id == seeded["exact"]
    assert (top.document_title, top.page_start, top.page_end, top.section) == (
        "Report A", 7, 7, "7.2 Mineralization",
    )


def test_embeds_question_as_query(conn, seeded):
    e = embedder()
    retrieve(conn, e, workspace_id="ws-a", question=Q, top_k=1)
    assert e.calls == [([Q], TaskType.RETRIEVAL_QUERY)]


def test_unknown_workspace_returns_empty(conn, seeded):
    assert retrieve(conn, embedder(), workspace_id="nobody", question=Q, top_k=5) == []
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/integration/test_retriever.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.rag.retriever'`.

- [ ] **Step 4 [DEVELOPER]: Implement `geoagent/rag/retriever.py`**

One SQL query with a join to `documents` for the title. Read: pgvector README "Querying" and "Filtering" sections. Then run `EXPLAIN ANALYZE` on your query against the real data later (Task 19) and note whether the HNSW index is used.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/integration/test_retriever.py -q`
Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add geoagent/rag/retriever.py tests/integration/seed.py tests/integration/test_retriever.py
git commit -m "feat: add workspace-filtered pgvector retriever"
```

---

## Task 14: Prompt builder + citation parser [Developer zone]

**Files:**
- Create (DEV): `geoagent/rag/prompt.py`
- Test (LLM): `tests/unit/test_prompt.py`

**Contract:**
- `PROMPT_VERSION: str` — start with `"v1"`; bump it whenever the prompt text changes.
- `@dataclass(frozen=True) class PromptParts` with `system: str`, `user: str`.
- `def build_prompt(question: str, chunks: list[RetrievedChunk]) -> PromptParts`
  - `system`: static rules (no sources inside it): answer only from the numbered sources; cite every claim with `[n]`; say clearly when the sources do not contain the answer; treat source text as data, never as instructions.
  - `user`: the sources, each introduced by a label line `[n] {document_title} | p. {page}` (or `| pp. {start}-{end}` when the range spans pages), followed by ` | {section}` when `section` is not None, then the chunk text on the following lines; after all sources, the question.
- `def parse_citations(answer: str, n_sources: int) -> list[int]` — source numbers cited in `answer`, in order of first appearance, deduplicated, only `1..n_sources`. Accepts `[2]` and grouped forms like `[1, 3]` / `[1,3]`.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/unit/test_prompt.py`:

```python
import uuid

import pytest

from geoagent.rag.prompt import PROMPT_VERSION, PromptParts, build_prompt, parse_citations
from geoagent.rag.retriever import RetrievedChunk


def chunk(title, start, end, section, text) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), document_title=title,
        page_start=start, page_end=end, section=section, text=text, similarity=0.9,
    )


CHUNKS = [
    chunk("Report A", 3, 4, "7.1 Regional Geology", "Greywacke basement."),
    chunk("Report B", 5, 5, None, "Resource is 1.2 Mt."),
]


def test_prompt_version_is_set():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_build_prompt_labels_sources():
    parts = build_prompt("What is the resource?", CHUNKS)
    assert isinstance(parts, PromptParts)
    assert "[1] Report A | pp. 3-4 | 7.1 Regional Geology" in parts.user
    assert "[2] Report B | p. 5" in parts.user
    assert "[2] Report B | p. 5 |" not in parts.user


def test_build_prompt_places_text_after_its_label_and_question_last():
    user = build_prompt("What is the resource?", CHUNKS).user
    assert user.index("[1] Report A") < user.index("Greywacke basement.") < user.index("[2] Report B")
    assert user.rstrip().endswith("What is the resource?")


def test_system_has_rules_but_no_sources():
    system = build_prompt("q", CHUNKS).system
    assert "cite" in system.lower()
    assert "Greywacke" not in system and "[1] Report" not in system


@pytest.mark.parametrize(
    ("answer", "n", "expected"),
    [
        ("Gold [1] and silver [3][2]. Also [1].", 3, [1, 3, 2]),
        ("Out of range [0] [4] but [2] ok.", 3, [2]),
        ("Grouped [1, 3] and [2,1].", 3, [1, 3, 2]),
        ("No citations at all.", 3, []),
        ("Year [2023] is not a citation.", 3, []),
    ],
)
def test_parse_citations(answer, n, expected):
    assert parse_citations(answer, n) == expected
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_prompt.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.rag.prompt'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/rag/prompt.py`**

Write the system rules in your own words. Think about: where in the prompt should the question go and why ("lost in the middle")? What could a malicious sentence inside a PDF do to this prompt, and which rule mitigates it (only partially)?

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_prompt.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add geoagent/rag/prompt.py tests/unit/test_prompt.py
git commit -m "feat: add grounded prompt builder and citation parser"
```

---

## Task 15: Answer orchestration [Developer zone]

> **Amendment — schema v2 (apply before executing this task):** `workspace_id` is typed `uuid.UUID` (the unit tests use fakes, so any value works there).

**Files:**
- Create (DEV): `geoagent/rag/answer.py`
- Test (LLM): `tests/unit/test_answer.py`

**Contract:**
- `NOT_FOUND_ANSWER = "I could not find this in the documents."`
- `@dataclass(frozen=True) class Citation` with `n: int`, `document_title: str`, `page_start: int`, `page_end: int`, `section: str | None`, `snippet: str` (first 200 characters of the chunk text).
- `@dataclass(frozen=True) class AskResponse` with `answer: str`, `found: bool`, `citations: list[Citation]`, `model: str | None`, `prompt_version: str`, `latency_ms: int`, `tokens_in: int`, `tokens_out: int`.
- `def answer_question(conn, embedder, llm, *, workspace_id: str, question: str, top_k: int, min_similarity: float, retrieve_fn=retrieve) -> AskResponse`
  1. `chunks = retrieve_fn(conn, embedder, workspace_id=workspace_id, question=question, top_k=top_k)`.
  2. Keep only chunks with `similarity >= min_similarity`. If none remain → return `AskResponse(answer=NOT_FOUND_ANSWER, found=False, citations=[], model=None, prompt_version=PROMPT_VERSION, latency_ms=..., tokens_in=0, tokens_out=0)` **without calling the LLM**.
  3. `build_prompt(question, kept)` → `llm.generate(parts.system, parts.user)`; provider exceptions propagate unchanged.
  4. `parse_citations(gen.text, len(kept))` → `Citation` for each, `n` matching the source number, fields from `kept[n-1]`.
  5. `latency_ms` = wall-clock milliseconds for the whole call (`time.perf_counter`), as an `int`.

- [ ] **Step 1 [LLM]: Write the failing tests**

`tests/unit/test_answer.py`:

```python
import uuid

import pytest

from geoagent.providers.errors import ProviderTimeout
from geoagent.rag.answer import NOT_FOUND_ANSWER, AskResponse, Citation, answer_question
from geoagent.rag.prompt import PROMPT_VERSION
from geoagent.rag.retriever import RetrievedChunk
from tests.fakes import FakeEmbedder, FakeLLM


def chunk(text: str, similarity: float, title: str = "Report A", page: int = 3) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), document_title=title,
        page_start=page, page_end=page, section="7.2 Mineralization", text=text,
        similarity=similarity,
    )


def fixed_retrieve(chunks):
    calls = []

    def retrieve_fn(conn, embedder, *, workspace_id, question, top_k):
        calls.append((workspace_id, question, top_k))
        return chunks

    retrieve_fn.calls = calls
    return retrieve_fn


def ask(llm, chunks, min_similarity=0.5):
    return answer_question(
        None, FakeEmbedder(), llm, workspace_id="ws", question="Grade?", top_k=4,
        min_similarity=min_similarity, retrieve_fn=fixed_retrieve(chunks),
    )


def test_not_found_short_circuits_without_llm():
    llm = FakeLLM()
    r = ask(llm, [chunk("weak", 0.2)])
    assert isinstance(r, AskResponse)
    assert (r.found, r.answer, r.citations, r.model) == (False, NOT_FOUND_ANSWER, [], None)
    assert (r.tokens_in, r.tokens_out) == (0, 0)
    assert r.prompt_version == PROMPT_VERSION
    assert llm.calls == []


def test_answer_with_valid_citations_only():
    long_text = "Gold grade is 2.3 g/t. " * 20
    llm = FakeLLM(text="It is 2.3 g/t [1]. Unrelated claim [7].", model="m1")
    r = ask(llm, [chunk(long_text, 0.9)])
    assert r.found is True
    assert r.answer == "It is 2.3 g/t [1]. Unrelated claim [7]."
    assert r.citations == [
        Citation(n=1, document_title="Report A", page_start=3, page_end=3,
                 section="7.2 Mineralization", snippet=long_text[:200])
    ]
    assert (r.model, r.tokens_in, r.tokens_out) == ("m1", 10, 5)


def test_low_similarity_chunks_are_not_sent_to_llm():
    llm = FakeLLM(text="ok [1]")
    ask(llm, [chunk("STRONG-TEXT", 0.8), chunk("WEAK-TEXT", 0.1)])
    _, user = llm.calls[0]
    assert "STRONG-TEXT" in user and "WEAK-TEXT" not in user


def test_citation_numbers_follow_kept_order():
    llm = FakeLLM(text="See [2].")
    r = ask(llm, [chunk("a", 0.9, title="A"), chunk("b", 0.8, title="B")])
    assert [(c.n, c.document_title) for c in r.citations] == [(2, "B")]


def test_passes_arguments_to_retriever():
    retrieve_fn = fixed_retrieve([])
    answer_question(None, FakeEmbedder(), FakeLLM(), workspace_id="ws-9", question="Q?", top_k=6,
                    min_similarity=0.5, retrieve_fn=retrieve_fn)
    assert retrieve_fn.calls == [("ws-9", "Q?", 6)]


def test_provider_errors_propagate():
    with pytest.raises(ProviderTimeout):
        ask(FakeLLM(fail_with=ProviderTimeout("slow")), [chunk("a", 0.9)])


def test_latency_is_non_negative_int():
    r = ask(FakeLLM(text="x [1]"), [chunk("a", 0.9)])
    assert isinstance(r.latency_ms, int) and r.latency_ms >= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_answer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.rag.answer'`.

- [ ] **Step 3 [DEVELOPER]: Implement `geoagent/rag/answer.py`**

Question to answer for yourself: why is injecting `retrieve_fn` (dependency injection) better here than mocking with `monkeypatch`? What would you log from this function to debug a bad answer later (Sprint 3 will turn this into tracing)?

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_answer.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add geoagent/rag/answer.py tests/unit/test_answer.py
git commit -m "feat: add answer orchestration with not-found short-circuit"
```

---

## Task 16: Logging, wiring, API [LLM zone]

> **Amendment — provider wiring (Task 7/8 reviews):** `OllamaProvider(..., num_ctx=...)` — pass all args by keyword; add setting `llm_num_ctx: int = 8192`. google-genai has **no timeout by default** (`timeout=None`): build **both** the embedding and the generation clients with `types.HttpOptions(timeout=<ms>)`. Do **not** set SDK `retry_options` on the embedding client (`GeminiEmbeddings` retries itself; stacking multiplies waits); generation may use modest SDK retries (e.g. 3 attempts) or none. Close the Ollama provider's client on app shutdown (FastAPI lifespan).

> **Amendment — schema v2 (apply before executing this task):** The API takes the workspace **slug** at the edge and resolves it once with `get_workspace_id` (`AskRequest.workspace: str`, `GET /documents?workspace=`, form field `workspace` on upload). Add an exception handler `WorkspaceNotFound` → 404 `{"error": "workspace_not_found"}`. Upload key: `incoming/{workspace_uuid}/{uuid4}.pdf` — **never put the user's filename in the key** (Task 5 review: on Windows, names differing only by case or trailing dots/spaces collapse to one file; GCS keeps them distinct); keep the original filename as metadata in the response/log. `BlobNotFound` and `ValueError` from the blob store map to 404 / 422. Unit tests monkeypatch `main.get_workspace_id` (the connection is faked); the integration test seeds with `ensure_workspace`.

**Files:**
- Create: `geoagent/logs.py`, `geoagent/wiring.py`, `geoagent/api/schemas.py`, `geoagent/api/main.py`
- Test: `tests/unit/test_logs.py`, `tests/unit/test_api.py`, `tests/integration/test_api_ask.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_logs.py`:

```python
import json
import logging

from geoagent.logs import JsonFormatter, request_id_var


def test_json_formatter_includes_request_id_and_extras():
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord("geo", logging.INFO, __file__, 1, "asked %s", ("q",), None)
        record.workspace_id = "ws"
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "asked q"
    assert payload["severity"] == "INFO"
    assert payload["request_id"] == "req-123"
    assert payload["workspace_id"] == "ws"
```

`tests/unit/test_api.py`:

```python
import psycopg
import pytest
from fastapi.testclient import TestClient

from geoagent import wiring
from geoagent.api import main
from geoagent.blobstore.local import LocalFsBlobStore
from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.rag.answer import AskResponse
from tests.fakes import FakeEmbedder, FakeLLM


def fake_conn():
    yield None


@pytest.fixture
def client(tmp_path):
    app = main.create_app()
    app.dependency_overrides[main.get_conn] = fake_conn
    app.dependency_overrides[wiring.get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[wiring.get_llm] = lambda: FakeLLM()
    app.dependency_overrides[wiring.get_blobstore] = lambda: LocalFsBlobStore(tmp_path, "raw")
    return TestClient(app, raise_server_exceptions=False)


def ok_response(**kw) -> AskResponse:
    base = dict(answer="a [1]", found=True, citations=[], model="m", prompt_version="v1",
                latency_ms=5, tokens_in=1, tokens_out=1)
    return AskResponse(**{**base, **kw})


def test_healthz_sets_request_id_header(client):
    resp = client.get("/healthz", headers={"x-request-id": "abc"})
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}
    assert resp.headers["x-request-id"] == "abc"


def test_ask_returns_answer(client, monkeypatch):
    seen = {}

    def fake_answer(conn, embedder, llm, **kw):
        seen.update(kw)
        return ok_response()

    monkeypatch.setattr(main, "answer_question", fake_answer)
    resp = client.post("/ask", json={"workspace_id": "ws", "question": "Grade?", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "a [1]"
    assert seen["workspace_id"] == "ws" and seen["top_k"] == 3


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (ProviderTimeout("slow"), 504, "llm_timeout"),
        (ProviderUnavailable("down"), 503, "provider_unavailable"),
        (ProviderError("bad"), 502, "provider_error"),
        (RuntimeError("bug"), 500, "internal_error"),
    ],
)
def test_ask_error_mapping(client, monkeypatch, exc, status, code):
    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(main, "answer_question", boom)
    resp = client.post("/ask", json={"workspace_id": "ws", "question": "Grade?"})
    assert resp.status_code == status
    body = resp.json()
    assert body["error"] == code and body["request_id"]


def test_database_unavailable_returns_503_with_retry_after(client):
    def database_down():
        raise psycopg.OperationalError("connection to server at 10.0.0.5 failed")

    client.app.dependency_overrides[main.get_conn] = database_down
    resp = client.post("/ask", json={"workspace_id": "ws", "question": "Grade?"})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "10"
    assert resp.json()["error"] == "database_unavailable"
    assert "10.0.0.5" not in resp.text


def test_ask_validation_error(client):
    resp = client.post("/ask", json={"workspace_id": "ws", "question": ""})
    assert resp.status_code == 422


def test_upload_document_writes_to_incoming(client, tmp_path):
    resp = client.post(
        "/documents",
        data={"workspace_id": "ws"},
        files={"file": ("report.pdf", b"%PDF-1.7 x", "application/pdf")},
    )
    assert resp.status_code == 202
    assert resp.json()["key"] == "incoming/ws/report.pdf"
    assert (tmp_path / "raw" / "incoming" / "ws" / "report.pdf").read_bytes() == b"%PDF-1.7 x"


def test_upload_rejects_non_pdf(client):
    resp = client.post(
        "/documents", data={"workspace_id": "ws"}, files={"file": ("a.txt", b"x", "text/plain")}
    )
    assert resp.status_code == 422
```

`tests/integration/test_api_ask.py`:

```python
import pytest
from fastapi.testclient import TestClient

from geoagent import wiring
from geoagent.api import main
from tests.fakes import FakeEmbedder, FakeLLM, basis_vector
from tests.integration.seed import seed_chunk

pytestmark = pytest.mark.integration


def test_ask_end_to_end_against_postgres(conn):
    seed_chunk(conn, workspace_id="ws", text="Indicated 1.2 Mt at 2.3 g/t Au.",
               vector=basis_vector(0), document_title="Kiwi Ridge", page=4,
               section="14 MINERAL RESOURCE ESTIMATES")
    app = main.create_app()

    def use_test_conn():
        yield conn

    app.dependency_overrides[main.get_conn] = use_test_conn
    app.dependency_overrides[wiring.get_embedder] = lambda: FakeEmbedder(
        fixed={"What is the indicated resource?": basis_vector(0)}
    )
    app.dependency_overrides[wiring.get_llm] = lambda: FakeLLM(text="1.2 Mt at 2.3 g/t Au [1].")
    resp = TestClient(app).post(
        "/ask", json={"workspace_id": "ws", "question": "What is the indicated resource?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["citations"][0]["document_title"] == "Kiwi Ridge"
    assert body["citations"][0]["page_start"] == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_logs.py tests/unit/test_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.logs'`.

- [ ] **Step 3: Implement `geoagent/logs.py`**

```python
import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; `severity` is understood by Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
```

- [ ] **Step 4: Implement `geoagent/wiring.py`**

```python
"""Build concrete providers from Settings. The only module that knows about every backend."""

from functools import lru_cache

import psycopg
from google import genai
from google.cloud import storage
from google.genai import types

from geoagent.blobstore.base import BlobStore
from geoagent.blobstore.gcs import GcsBlobStore
from geoagent.blobstore.local import LocalFsBlobStore
from geoagent.config import get_settings
from geoagent.db.connection import connect
from geoagent.providers.gemini import GeminiEmbeddings, VertexGeminiProvider
from geoagent.providers.ollama import OllamaProvider
from geoagent.providers.types import EmbeddingProvider, LLMProvider


def open_connection(*, statement_timeout_ms: int | None = None) -> psycopg.Connection:
    return connect(get_settings(), statement_timeout_ms=statement_timeout_ms)


@lru_cache
def get_embedder() -> EmbeddingProvider:
    s = get_settings()
    if s.gemini_api_key is not None:
        client = genai.Client(api_key=s.gemini_api_key.get_secret_value())
    else:
        client = genai.Client(
            vertexai=True, project=s.google_cloud_project, location=s.embedding_location
        )
    return GeminiEmbeddings(
        client=client, model=s.embedding_model, dim=s.embedding_dim,
        batch_size=s.embedding_batch_size,
    )


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "ollama":
        return OllamaProvider(base_url=s.ollama_base_url, model=s.llm_model, timeout_s=s.llm_timeout_s)
    client = genai.Client(
        vertexai=True,
        project=s.google_cloud_project,
        location=s.google_cloud_location,
        http_options=types.HttpOptions(timeout=int(s.llm_timeout_s * 1000)),
    )
    return VertexGeminiProvider(client=client, model=s.llm_model)


@lru_cache
def get_blobstore() -> BlobStore:
    s = get_settings()
    if s.blob_store == "gcs":
        return GcsBlobStore(client=storage.Client(), bucket=s.blob_bucket)
    return LocalFsBlobStore(root=s.blob_root, bucket=s.blob_bucket)
```

- [ ] **Step 5: Implement `geoagent/api/schemas.py`**

```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class CitationOut(BaseModel):
    n: int
    document_title: str
    page_start: int
    page_end: int
    section: str | None
    snippet: str


class AskResponseOut(BaseModel):
    answer: str
    found: bool
    citations: list[CitationOut]
    model: str | None
    prompt_version: str
    latency_ms: int
    tokens_in: int
    tokens_out: int


class DocumentOut(BaseModel):
    id: UUID
    title: str
    status: str
    page_count: int | None
    error: str | None
    created_at: datetime


class UploadOut(BaseModel):
    key: str
    uri: str
```

- [ ] **Step 6: Implement `geoagent/api/main.py`**

```python
import dataclasses
import logging
import uuid
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from geoagent import wiring
from geoagent.api.schemas import AskRequest, AskResponseOut, DocumentOut, UploadOut
from geoagent.blobstore.base import BlobStore
from geoagent.config import get_settings
from geoagent.logs import configure_logging, request_id_var
from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import EmbeddingProvider, LLMProvider
from geoagent.rag.answer import answer_question

log = logging.getLogger("geoagent.api")


def get_conn() -> Iterator[psycopg.Connection]:
    conn = wiring.open_connection()
    try:
        yield conn
    finally:
        conn.close()


def _error(status: int, code: str, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": code, "detail": str(exc), "request_id": request_id_var.get()},
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="GeoAgent", version="0.1.0")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            response = JSONResponse(
                status_code=500, content={"error": "internal_error", "request_id": rid}
            )
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(ProviderTimeout)
    async def on_timeout(request: Request, exc: ProviderTimeout):
        return _error(504, "llm_timeout", exc)

    @app.exception_handler(ProviderUnavailable)
    async def on_unavailable(request: Request, exc: ProviderUnavailable):
        return _error(503, "provider_unavailable", exc)

    @app.exception_handler(ProviderError)
    async def on_provider_error(request: Request, exc: ProviderError):
        return _error(502, "provider_error", exc)

    @app.exception_handler(psycopg.OperationalError)
    async def on_database_unavailable(request: Request, exc: psycopg.OperationalError):
        # Generic detail on purpose: driver messages reveal internal hosts and ports.
        log.warning("database unavailable", extra={"error": type(exc).__name__})
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "10"},
            content={
                "error": "database_unavailable",
                "detail": "database temporarily unavailable",
                "request_id": request_id_var.get(),
            },
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponseOut)
    def ask(
        body: AskRequest,
        conn: Annotated[psycopg.Connection, Depends(get_conn)],
        embedder: Annotated[EmbeddingProvider, Depends(wiring.get_embedder)],
        llm: Annotated[LLMProvider, Depends(wiring.get_llm)],
    ):
        result = answer_question(
            conn, embedder, llm,
            workspace_id=body.workspace_id,
            question=body.question,
            top_k=body.top_k or settings.top_k,
            min_similarity=settings.min_similarity,
        )
        log.info(
            "ask",
            extra={
                "workspace_id": body.workspace_id,
                "found": result.found,
                "model": result.model,
                "prompt_version": result.prompt_version,
                "latency_ms": result.latency_ms,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "n_citations": len(result.citations),
            },
        )
        return dataclasses.asdict(result)

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(
        workspace_id: str, conn: Annotated[psycopg.Connection, Depends(get_conn)]
    ):
        rows = conn.execute(
            "SELECT id, title, status, page_count, error, created_at FROM documents "
            "WHERE workspace_id = %s ORDER BY created_at",
            (workspace_id,),
        ).fetchall()
        return [
            DocumentOut(id=r[0], title=r[1], status=r[2], page_count=r[3], error=r[4], created_at=r[5])
            for r in rows
        ]

    @app.post("/documents", status_code=202, response_model=UploadOut)
    def upload_document(
        workspace_id: Annotated[str, Form(min_length=1)],
        file: Annotated[UploadFile, File()],
        blobs: Annotated[BlobStore, Depends(wiring.get_blobstore)],
    ):
        name = PurePosixPath(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=422, detail="only .pdf files are accepted")
        key = f"incoming/{workspace_id}/{name}"
        uri = blobs.put(key, file.file.read(), "application/pdf")
        return UploadOut(key=key, uri=uri)

    return app


app = create_app()
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/unit/test_logs.py tests/unit/test_api.py -q`
Expected: `11 passed`.

Run: `uv run pytest tests/integration/test_api_ask.py -q`
Expected: `1 passed`.

- [ ] **Step 8: Developer review checklist** — why does `POST /documents` only *upload* (202) instead of ingesting synchronously? Why is a new DB connection per request acceptable now but not at scale (connection pooling is a Sprint 3+ topic)? Where does `request_id` end up in Cloud Logging?

- [ ] **Step 9: Commit**

```bash
git add geoagent/logs.py geoagent/wiring.py geoagent/api tests/unit/test_logs.py tests/unit/test_api.py tests/integration/test_api_ask.py
git commit -m "feat: add FastAPI app, provider wiring and JSON logging"
```

---

## Task 17: CLI, smoke loader, report fetcher [LLM zone]

> **Amendment — schema v2 (apply before executing this task):** `--workspace` takes a slug, resolved with `get_workspace_id` (clear error if unknown). Add a `workspace` sub-command group: `geoagent workspace create SLUG --name NAME` and `geoagent workspace list`.

**Files:**
- Create: `geoagent/smoke.py`, `geoagent/cli.py`, `scripts/fetch_reports.py`
- Test: `tests/unit/test_smoke_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_smoke_loader.py`:

```python
import json

import pytest

from geoagent.smoke import SmokeItem, load_smoke


def write(tmp_path, lines):
    p = tmp_path / "smoke.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def item(**kw):
    base = {"id": "q1", "question": "Q?", "expected_answer": "A", "expected_document": "macraes",
            "category": "exact_term"}
    return json.dumps({**base, **kw})


def test_loads_items_and_skips_blank_lines(tmp_path):
    items = load_smoke(write(tmp_path, [item(), "", item(id="q2", category="not_in_corpus")]))
    assert [i.id for i in items] == ["q1", "q2"]
    assert isinstance(items[0], SmokeItem)
    assert items[0].expected_pages is None


def test_rejects_unknown_category_with_line_number(tmp_path):
    with pytest.raises(ValueError, match="line 2"):
        load_smoke(write(tmp_path, [item(), item(id="q2", category="vibes")]))


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_smoke(write(tmp_path, [item(), item()]))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_smoke_loader.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'geoagent.smoke'`.

- [ ] **Step 3: Implement `geoagent/smoke.py`**

```python
import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = {"exact_term", "conceptual", "cross_document_trap", "not_in_corpus"}


@dataclass(frozen=True)
class SmokeItem:
    id: str
    question: str
    expected_answer: str
    expected_document: str | None
    category: str
    expected_pages: list[int] | None = None


def load_smoke(path: Path) -> list[SmokeItem]:
    items: list[SmokeItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("category") not in CATEGORIES:
            raise ValueError(f"line {lineno}: category must be one of {sorted(CATEGORIES)}")
        if raw["id"] in seen:
            raise ValueError(f"line {lineno}: duplicate id {raw['id']!r}")
        seen.add(raw["id"])
        items.append(
            SmokeItem(
                id=raw["id"],
                question=raw["question"],
                expected_answer=raw["expected_answer"],
                expected_document=raw.get("expected_document"),
                category=raw["category"],
                expected_pages=raw.get("expected_pages"),
            )
        )
    return items
```

- [ ] **Step 4: Implement `geoagent/cli.py`**

```python
from pathlib import Path
from typing import Annotated

import typer

from geoagent import wiring
from geoagent.config import get_settings
from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations
from geoagent.ingest.pipeline import IngestResult, ingest_bytes, ingest_from_blob
from geoagent.logs import configure_logging
from geoagent.rag.answer import AskResponse, answer_question
from geoagent.smoke import load_smoke

app = typer.Typer(help="GeoAgent command-line interface", no_args_is_help=True)

# Batch work (migrations, bulk inserts) gets a longer statement timeout than API requests.
BATCH_STATEMENT_TIMEOUT_MS = 120_000


@app.callback()
def main() -> None:
    configure_logging(get_settings().log_level)


@app.command()
def migrate() -> None:
    """Apply pending SQL migrations."""
    with connect(
        get_settings(), statement_timeout_ms=BATCH_STATEMENT_TIMEOUT_MS, register_vector_type=False
    ) as conn:
        applied = apply_migrations(conn)
    typer.echo(f"applied: {', '.join(applied) if applied else 'nothing (up to date)'}")


def _report(results: list[IngestResult]) -> None:
    for r in results:
        flag = "skipped" if r.skipped else r.status
        typer.echo(f"{flag:8} {r.title}  {r.document_id}  {r.error or ''}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=1)


@app.command()
def ingest(
    workspace: Annotated[str, typer.Option(help="Workspace id, e.g. nz-gold")],
    paths: Annotated[list[Path] | None, typer.Argument(help="Local PDF files")] = None,
    from_blob: Annotated[str | None, typer.Option(help="Blob prefix, e.g. incoming/nz-gold/")] = None,
) -> None:
    """Ingest PDFs from local paths and/or a blob-store prefix."""
    if not paths and not from_blob:
        raise typer.BadParameter("give PDF paths and/or --from-blob PREFIX")
    blobs, embedder = wiring.get_blobstore(), wiring.get_embedder()
    results: list[IngestResult] = []
    with wiring.open_connection(statement_timeout_ms=BATCH_STATEMENT_TIMEOUT_MS) as conn:
        for path in paths or []:
            results.append(
                ingest_bytes(conn, blobs, embedder, workspace_id=workspace,
                             data=path.read_bytes(), title=path.stem)
            )
        if from_blob:
            results.extend(
                ingest_from_blob(conn, blobs, embedder, workspace_id=workspace, prefix=from_blob)
            )
    _report(results)


def _ask(conn, workspace: str, question: str, top_k: int | None) -> AskResponse:
    s = get_settings()
    return answer_question(
        conn, wiring.get_embedder(), wiring.get_llm(), workspace_id=workspace,
        question=question, top_k=top_k or s.top_k, min_similarity=s.min_similarity,
    )


def _print_answer(r: AskResponse) -> None:
    typer.echo(r.answer)
    for c in r.citations:
        pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"pp. {c.page_start}-{c.page_end}"
        typer.echo(f"  [{c.n}] {c.document_title} | {pages} | {c.section or '-'}")
    typer.echo(f"  ({r.model}, {r.prompt_version}, {r.latency_ms} ms, "
               f"{r.tokens_in}+{r.tokens_out} tokens)")


@app.command()
def ask(
    question: str,
    workspace: Annotated[str, typer.Option()] = "nz-gold",
    top_k: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Ask a question and print the cited answer."""
    with wiring.open_connection() as conn:
        _print_answer(_ask(conn, workspace, question, top_k))


@app.command()
def smoke(
    path: Annotated[Path, typer.Argument()] = Path("evals/smoke.jsonl"),
    workspace: Annotated[str, typer.Option()] = "nz-gold",
) -> None:
    """Run the smoke set and print answers next to expectations for manual review."""
    items = load_smoke(path)
    with wiring.open_connection() as conn:
        for item in items:
            typer.echo(f"\n=== {item.id} [{item.category}] {item.question}")
            typer.echo(f"expected: {item.expected_answer} ({item.expected_document})")
            _print_answer(_ask(conn, workspace, item.question, None))


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Implement `scripts/fetch_reports.py`**

```python
"""Download the public NI 43-101 reports into data/ (git-ignored; reports are copyrighted)."""

import hashlib
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REPORTS = {
    "waihi-ni43-101.pdf": "https://assets.oceanagold.com/documents/Reports/Technical-Reports/"
    "WaihiDistrictNI43101TechnicalReport.pdf",
    "macraes-ni43-101.pdf": "https://assets.oceanagold.com/documents/Reports/Technical-Reports/"
    "Macraes-Operation-NI-43-101-Technical-Report.pdf",
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for block in resp.iter_bytes():
                fh.write(block)
    tmp.replace(dest)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in REPORTS.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"exists   {name}")
        else:
            print(f"fetching {name}")
            download(url, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"  {dest.stat().st_size / 1e6:.1f} MB  sha256={digest[:16]}…")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and CLI help**

Run: `uv run pytest tests/unit/test_smoke_loader.py -q`
Expected: `3 passed`.

Run: `uv run geoagent --help`
Expected: lists commands `migrate`, `ingest`, `ask`, `smoke`.

- [ ] **Step 7: Fetch the reports**

Run: `uv run python scripts/fetch_reports.py`
Expected: two `fetching` lines followed by sizes; `data/` now holds `waihi-ni43-101.pdf` and `macraes-ni43-101.pdf`. Confirm `git status` does not list `data/`.

- [ ] **Step 8: Commit**

```bash
git add geoagent/smoke.py geoagent/cli.py scripts/fetch_reports.py tests/unit/test_smoke_loader.py
git commit -m "feat: add CLI (migrate/ingest/ask/smoke) and report fetcher"
```

---

## Task 18: Local LLM host, Dockerfile and compose API [LLM zone]

**Files:**
- Create: `docs/local-llm.md`, `Dockerfile`
- Modify: `docker-compose.yml` (add `api` service)

- [ ] **Step 1: Create `docs/local-llm.md`**

````markdown
# Local LLM host (Ollama)

Ollama runs **outside** Docker Compose, on any GPU machine on the LAN. The API reaches it via
`OLLAMA_BASE_URL`. Ollama has **no authentication** — keep port 11434 on the LAN only, never
expose it to the internet.

## Windows + NVIDIA GPU (the MVP host)

Install Ollama for Windows (https://ollama.com/download), then:

```bash
ollama pull qwen3:8b
curl http://127.0.0.1:11434/api/tags
```

The compose `api` container reaches it at `http://host.docker.internal:11434`. If that request
fails, Ollama is listening on loopback only: set the user environment variable
`OLLAMA_HOST=0.0.0.0`, restart Ollama, and allow port 11434 on private networks only.

## Linux + AMD GPU (ROCm, RDNA4 needs the ROCm 7 driver)

Podman (works on immutable/atomic distros without layering packages):

```bash
podman run -d --name ollama --restart=always \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt label=disable \
  -p 11434:11434 -v ollama:/root/.ollama \
  docker.io/ollama/ollama:rocm
podman exec ollama ollama pull qwen3:14b
```

Open the port on the LAN zone if firewalld is active:

```bash
sudo firewall-cmd --add-port=11434/tcp --permanent && sudo firewall-cmd --reload
```

If the GPU is not detected, check `podman logs ollama` for the detected `gfx` target; as a last
resort add `-e HSA_OVERRIDE_GFX_VERSION=<version>` (a workaround, not a fix).

## macOS (Apple Silicon)

Docker on macOS cannot use the Metal GPU, so run Ollama natively:

```bash
brew install ollama
OLLAMA_HOST=0.0.0.0 ollama serve
ollama pull gemma3:27b
```

## Verify from the dev machine

```bash
curl http://<gpu-host>:11434/api/tags
```

Expected: JSON listing the pulled models. Then set `OLLAMA_BASE_URL` and `LLM_MODEL` in `.env`.
````

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY geoagent ./geoagent
RUN uv sync --frozen --no-dev \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn geoagent.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

- [ ] **Step 3: Add the `api` service to `docker-compose.yml`**

Replace the file with:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${DB_USER:?set DB_USER in .env}
      POSTGRES_PASSWORD: ${DB_PASSWORD:?set DB_PASSWORD in .env}
      POSTGRES_DB: ${DB_NAME:-geoagent}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 20

  api:
    build: .
    env_file: .env
    environment:
      DB_HOST: postgres
      BLOB_ROOT: /app/blobdata
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    volumes:
      - ./blobdata:/app/blobdata
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 4: Build and run**

Prerequisite: `.env` has `DB_USER`, `DB_PASSWORD` and `GEMINI_API_KEY` filled in; Ollama answers the `curl` check from `docs/local-llm.md`.

Run: `docker compose up -d --build` then `curl http://127.0.0.1:8080/healthz`
Expected: `{"status":"ok"}`.

Run: `uv run geoagent migrate`
Expected: `applied: 001_init.sql` (or `nothing (up to date)`).

- [ ] **Step 5: Commit**

```bash
git add docs/local-llm.md Dockerfile docker-compose.yml
git commit -m "build: add Dockerfile, compose api service and local LLM host guide"
```

---

## Task 19: Real data, smoke set and first experiments [Developer zone]

> **Amendment — schema v2 (apply before executing this task):** First create the simulated tenants: `geoagent workspace create tenant-a --name "Tenant A"`, same for `tenant-b`, `tenant-c`. Ingest Waihi into `tenant-a` and Macraes into `tenant-b` (tenant-c gets a third public report later). Add an isolation check: a Macraes question asked as `tenant-a` must return the not-found answer.

**Files:**
- Create (DEV): `evals/smoke.jsonl`
- Modify (DEV): `docs/learning-log.md`

- [ ] **Step 1: Ingest the real reports**

Run: `uv run geoagent ingest --workspace nz-gold data/waihi-ni43-101.pdf data/macraes-ni43-101.pdf`
Expected: two `ready` lines. Then `curl "http://127.0.0.1:8080/documents?workspace_id=nz-gold"` lists both with `page_count`.

Inspect: `docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' -c "SELECT d.title, count(*), avg(c.token_count)::int FROM chunks c JOIN documents d ON d.id = c.document_id GROUP BY 1"`.

- [ ] **Step 2: Ask a first question**

Run: `uv run geoagent ask "What is the Mineral Resource estimate for Macraes?"`
Expected: an answer with `[n]` citations pointing at the Macraes report pages. Open the PDF at the cited pages and check.

- [ ] **Step 3 [DEVELOPER]: Write `evals/smoke.jsonl`** — about 30 lines, one JSON object per line:

```
{"id": "...", "question": "...", "expected_answer": "...", "expected_document": "waihi-ni43-101|macraes-ni43-101|null", "category": "exact_term|conceptual|cross_document_trap|not_in_corpus", "expected_pages": [optional list of ints]}
```

Read the reports and write the facts yourself. Aim for: ~10 `exact_term` (tonnages, grades, drillhole IDs, named faults/veins), ~8 `conceptual` (what controls mineralization, mining method, metallurgy), ~7 `cross_document_trap` (questions that only make sense for one of the two deposits, e.g. a grade asked without naming the deposit), ~5 `not_in_corpus` (answers absent from both reports).

- [ ] **Step 4: Run the smoke set and review**

Run: `uv run geoagent smoke`
Mark each answer correct / partially correct / wrong / wrongly-not-found in a scratch table; record the totals per category in `docs/learning-log.md`.

- [ ] **Step 5 [DEVELOPER]: Run the initial experiments** — for each, write the prediction in the log **first**, change one variable, re-ingest into a fresh workspace (e.g. `--workspace exp-chunk256`; add a temporary CLI option or edit defaults locally), re-run the smoke set, record the result:
1. Chunk size 256 / 512 / 1024 tokens.
2. Header/footer stripping on vs off.
3. Questions embedded as `RETRIEVAL_DOCUMENT` instead of `RETRIEVAL_QUERY`.
4. `top_k` 3 / 8 / 20.
5. `MIN_SIMILARITY` calibration: print similarity of the top chunk for `not_in_corpus` vs answerable questions; pick a threshold that separates them; update `.env`/defaults.

- [ ] **Step 6: Commit**

```bash
git add evals/smoke.jsonl docs/learning-log.md
git commit -m "eval: add smoke set and first learning-log experiments"
```

---

## Task 20: Cross-environment embedding check [LLM zone]

> **Status: superseded (2026-10-07).** The MVP no longer deploys to GCP. These tasks will be replaced by a cloud design document, local Terraform (Docker provider), and GCP Terraform that is validated but never applied. Kept for reference only; do not execute.

Verifies the spec's core assumption: AI Studio and Vertex AI return the same vectors for `gemini-embedding-001`, so one index serves both.

**Files:**
- Create: `scripts/compare_embeddings.py`

- [ ] **Step 1: Implement `scripts/compare_embeddings.py`**

```python
"""Embed the same texts via AI Studio (API key) and Vertex AI (ADC) and compare cosine similarity.

Usage: GEMINI_API_KEY=... uv run python scripts/compare_embeddings.py <gcp-project> [location]
Requires `gcloud auth application-default login` for the Vertex side.
"""

import os
import sys

from google import genai

from geoagent.providers.gemini import GeminiEmbeddings
from geoagent.providers.types import TaskType

TEXTS = [
    "Gold occurs in quartz-sulphide veins hosted by andesite.",
    "The Indicated Mineral Resource is 1.2 Mt at 2.3 g/t Au.",
]


def main() -> None:
    project = sys.argv[1]
    location = sys.argv[2] if len(sys.argv) > 2 else "us-central1"
    studio = GeminiEmbeddings(genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
                              "gemini-embedding-001", 768)
    vertex = GeminiEmbeddings(genai.Client(vertexai=True, project=project, location=location),
                              "gemini-embedding-001", 768)
    a = studio.embed(TEXTS, TaskType.RETRIEVAL_DOCUMENT)
    b = vertex.embed(TEXTS, TaskType.RETRIEVAL_DOCUMENT)
    for text, va, vb in zip(TEXTS, a, b):
        cos = sum(x * y for x, y in zip(va, vb))
        print(f"cos={cos:.6f}  {text[:50]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit** (run it in Task 23 once the GCP project exists)

```bash
git add scripts/compare_embeddings.py
git commit -m "chore: add AI Studio vs Vertex embedding comparison script"
```

---

## Task 21: Terraform bootstrap (state bucket) [Developer zone]

> **Status: superseded (2026-10-07).** The MVP no longer deploys to GCP. These tasks will be replaced by a cloud design document, local Terraform (Docker provider), and GCP Terraform that is validated but never applied. Kept for reference only; do not execute.

**Files:**
- Create (DEV): `infra/terraform/bootstrap/main.tf`

**Prerequisites (one-time, manual):** a GCP project with billing enabled; `gcloud` and `terraform` (≥ 1.6) installed; `gcloud auth login` and `gcloud auth application-default login`; `gcloud config set project <PROJECT_ID>`.

**Contract — `bootstrap/main.tf` must declare:**
- `terraform { required_providers { google = { source = "hashicorp/google", version = "~> 6.0" } } }` — check the current major version on the Terraform Registry and use it.
- `variable "project_id"` and `variable "region"` (default `"us-central1"`).
- `provider "google"` with `project` and `region` from the variables.
- `resource "google_storage_bucket" "tfstate"`: name `"${var.project_id}-tfstate"`, location `var.region`, `uniform_bucket_level_access = true`, `public_access_prevention = "enforced"`, `versioning { enabled = true }`, `force_destroy = false`.
- `output "state_bucket"` = the bucket name.
- Local state for bootstrap itself (no backend block). It is applied once and never destroyed.

- [ ] **Step 1 [DEVELOPER]: Write `bootstrap/main.tf`** — docs: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket. Why is state versioning important? Why does the state bucket live outside the stack it describes?

- [ ] **Step 2: Validate**

Run: `cd infra/terraform/bootstrap && terraform init && terraform fmt -check && terraform validate`
Expected: `Success! The configuration is valid.`

- [ ] **Step 3: Plan and apply**

Run: `terraform plan -var project_id=<PROJECT_ID>`
Expected: `Plan: 1 to add, 0 to change, 0 to destroy.`

Run: `terraform apply -var project_id=<PROJECT_ID>` → `yes`
Expected: `Apply complete! Resources: 1 added` and the `state_bucket` output.

- [ ] **Step 4: Commit** (state files are git-ignored)

```bash
git add infra/terraform/bootstrap/main.tf infra/terraform/bootstrap/.terraform.lock.hcl
git commit -m "infra: add terraform state bucket bootstrap"
```

---

## Task 22: Terraform main stack [scaffold: LLM zone; resources: Developer zone]

> **Status: superseded (2026-10-07).** The MVP no longer deploys to GCP. These tasks will be replaced by a cloud design document, local Terraform (Docker provider), and GCP Terraform that is validated but never applied. Kept for reference only; do not execute.

**Files:**
- Create (LLM): `infra/terraform/main/versions.tf`, `variables.tf`, `locals.tf`, `outputs.tf`
- Create (DEV): `apis.tf`, `artifact_registry.tf`, `storage.tf`, `sql.tf`, `secrets.tf`, `iam.tf`, `cloud_run.tf`, `budget.tf`

- [ ] **Step 1 [LLM]: Create `versions.tf`**

```hcl
terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  backend "gcs" {
    prefix = "geoagent/main"
    # bucket is passed at init time: terraform init -backend-config="bucket=<PROJECT_ID>-tfstate"
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}
```

(Use the same google provider major version chosen in Task 21.)

- [ ] **Step 2 [LLM]: Create `variables.tf`**

```hcl
variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type        = string
  description = "Full image URL, e.g. us-central1-docker.pkg.dev/<project>/geoagent/geoagent:v1"
  default     = ""
}

variable "llm_model" {
  type    = string
  default = "gemini-3.8-flash"
}

variable "workspace_id" {
  type    = string
  default = "nz-gold"
}

variable "billing_account_id" {
  type        = string
  description = "Billing account ID (gcloud billing accounts list)"
}

variable "budget_amount" {
  type    = number
  default = 20
}

variable "budget_currency" {
  type        = string
  default     = "USD"
  description = "Must match the billing account currency"
}
```

- [ ] **Step 3 [LLM]: Create `locals.tf`** (shared container environment for the service and the job)

```hcl
locals {
  app_env = {
    LLM_PROVIDER          = "vertex"
    LLM_MODEL             = var.llm_model
    GOOGLE_CLOUD_PROJECT  = var.project_id
    GOOGLE_CLOUD_LOCATION = "global"
    EMBEDDING_LOCATION    = var.region
    EMBEDDING_MODEL       = "gemini-embedding-001"
    EMBEDDING_DIM         = "768"
    EMBEDDING_BATCH_SIZE  = "32"
    BLOB_STORE            = "gcs"
    BLOB_BUCKET           = google_storage_bucket.raw.name
    LOG_LEVEL             = "INFO"
  }
}
```

- [ ] **Step 4 [LLM]: Create `outputs.tf`**

```hcl
output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "bucket" {
  value = google_storage_bucket.raw.name
}

output "image_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.geoagent.repository_id}"
}

output "sql_connection_name" {
  value = google_sql_database_instance.geoagent.connection_name
}
```

- [ ] **Step 5 [DEVELOPER]: Write the resource files to this contract**

Resource names below are referenced by `locals.tf`/`outputs.tf` — use them exactly.

| File | Resources (Terraform name) | Required settings |
|---|---|---|
| `apis.tf` | `google_project_service.services` (`for_each`) | `run`, `sqladmin`, `artifactregistry`, `secretmanager`, `aiplatform`, `storage`, `billingbudgets` (`*.googleapis.com`); `disable_on_destroy = false` |
| `artifact_registry.tf` | `google_artifact_registry_repository.geoagent` | `repository_id = "geoagent"`, `format = "DOCKER"`, `location = var.region` |
| `storage.tf` | `google_storage_bucket.raw` | name `"${var.project_id}-geoagent-raw"`, `uniform_bucket_level_access = true`, `public_access_prevention = "enforced"`, `force_destroy = true` |
| `sql.tf` | `google_sql_database_instance.geoagent`, `google_sql_database.geoagent`, `random_password.db`, `google_sql_user.geoagent` | `POSTGRES_16`; `settings { edition = "ENTERPRISE", tier = "db-f1-micro", availability_type = "ZONAL", disk_size = 10, ip_configuration { ipv4_enabled = true } }` with **no** `authorized_networks`; backups off; `deletion_protection = false`; database and user both named `geoagent`; password `length = 32`, `special = false` |
| `secrets.tf` | `google_secret_manager_secret.database_url`, `google_secret_manager_secret_version.database_url` | `secret_id = "geoagent-database-url"`, `replication { auto {} }`; value `postgresql://geoagent:<password>@/geoagent?host=/cloudsql/<connection_name>` |
| `iam.tf` | `google_service_account.runtime`; `google_project_iam_member` ×2; `google_storage_bucket_iam_member.runtime`; `google_secret_manager_secret_iam_member.runtime` | account id `geoagent-runtime`; project roles `roles/cloudsql.client`, `roles/aiplatform.user`; bucket role `roles/storage.objectUser`; secret role `roles/secretmanager.secretAccessor` |
| `cloud_run.tf` | `google_cloud_run_v2_service.api`, `google_cloud_run_v2_job.ingest` | both: `service_account` = runtime SA; `volumes { name = "cloudsql", cloud_sql_instance { instances = [connection_name] } }` mounted at `/cloudsql`; `image = var.image`; env from `local.app_env` (use a `dynamic "env"` block) plus `DATABASE_URL` from the secret (`value_source.secret_key_ref`, version `"latest"`); `deletion_protection = false`. Service: `scaling { min_instance_count = 0, max_instance_count = 2 }`, 1 CPU / 1Gi. **No** `allUsers` invoker binding (private). Job: `timeout = "3600s"`, `max_retries = 1`, 2Gi memory, `command = ["sh", "-c"]`, `args = ["geoagent migrate && geoagent ingest --workspace ${var.workspace_id} --from-blob incoming/${var.workspace_id}/"]` |
| `budget.tf` | `data.google_project.this`, `google_billing_budget.geoagent` | `billing_account = var.billing_account_id`; filter to `projects/${data.google_project.this.number}`; `specified_amount` in `var.budget_currency`/`var.budget_amount`; threshold rules at 0.5, 0.9, 1.0 |

Every resource that needs an API should `depends_on = [google_project_service.services]`. Docs: Terraform Registry `hashicorp/google` pages for each resource type. Questions: why is the service private, and how will you call it? Why does `DATABASE_URL` live in Secret Manager rather than a plain env var? What breaks if the runtime SA lacks `cloudsql.client`?

- [ ] **Step 6: Validate**

Run: `cd infra/terraform/main && terraform init -backend-config="bucket=<PROJECT_ID>-tfstate" && terraform fmt -check && terraform validate`
Expected: `Success! The configuration is valid.`

Run: `terraform plan -var project_id=<PROJECT_ID> -var billing_account_id=<BILLING_ID> -var image=placeholder`
Expected: plan lists the resources above with `0 to change, 0 to destroy` (exact add count depends on `for_each` size — check each expected resource appears). Do **not** apply yet.

- [ ] **Step 7: Commit**

```bash
git add infra/terraform/main/*.tf infra/terraform/main/.terraform.lock.hcl
git commit -m "infra: add main GCP stack (Cloud Run, Cloud SQL, GCS, Secret Manager, IAM, budget)"
```

---

## Task 23: Deploy runbook, cloud run-through, destroy [runbook: LLM zone; execution: Developer]

> **Status: superseded (2026-10-07).** The MVP no longer deploys to GCP. These tasks will be replaced by a cloud design document, local Terraform (Docker provider), and GCP Terraform that is validated but never applied. Kept for reference only; do not execute.

**Files:**
- Create: `docs/deploy.md`

- [ ] **Step 1 [LLM]: Create `docs/deploy.md`**

````markdown
# Deploy runbook (apply → test → destroy)

Variables used below:

```bash
export PROJECT_ID=<project>
export REGION=us-central1
export BILLING_ID=<billing account id>
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/geoagent/geoagent:v1
TF="-var project_id=$PROJECT_ID -var billing_account_id=$BILLING_ID -var image=$IMAGE"
```

## 1. Create APIs and the image repository first (the service needs an image to exist)

```bash
cd infra/terraform/main
terraform init -backend-config="bucket=$PROJECT_ID-tfstate"
terraform apply $TF -target=google_project_service.services -target=google_artifact_registry_repository.geoagent
```

## 2. Build and push the image (from the repo root)

```bash
gcloud auth configure-docker $REGION-docker.pkg.dev
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE
```

## 3. Apply the full stack (Cloud SQL creation takes ~10 minutes)

```bash
cd infra/terraform/main
terraform apply $TF
```

## 4. Upload the reports and run ingestion

```bash
BUCKET=$(terraform output -raw bucket)
gcloud storage cp ../../../data/*.pdf gs://$BUCKET/incoming/nz-gold/
gcloud run jobs execute geoagent-ingest --region $REGION --wait
```

Check logs: `gcloud logging read 'resource.type="cloud_run_job"' --limit 50 --format=json`.

## 5. Ask a question (the service is private: authenticate with an identity token)

```bash
URL=$(terraform output -raw api_url)
curl -s -X POST "$URL/ask" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"nz-gold","question":"What is the Mineral Resource estimate for Macraes?"}'
```

## 6. Destroy

```bash
terraform destroy $TF
```

Checklist — all must be empty:

```bash
gcloud sql instances list
gcloud run services list --region $REGION
gcloud run jobs list --region $REGION
gcloud artifacts repositories list --location $REGION
gcloud storage buckets list --filter="name~geoagent-raw"
```

The `-tfstate` bucket from `bootstrap/` intentionally remains.
````

- [ ] **Step 2 [DEVELOPER]: Embedding parity check**

Run: `GEMINI_API_KEY=<key> uv run python scripts/compare_embeddings.py <PROJECT_ID> us-central1`
Expected: `cos=` values ≥ 0.999 for both texts. If not, record it in the learning log — it would mean the local index cannot be reused in the cloud, and the spec's "one index everywhere" assumption must change.

- [ ] **Step 3 [DEVELOPER]: Execute runbook sections 1–5**

Expected at section 5: JSON with `"found": true`, citations to the Macraes report, `"model": "gemini-3.8-flash"`. If Vertex returns a model-not-found or location error, check model availability for `GOOGLE_CLOUD_LOCATION=global` / `EMBEDDING_LOCATION` in the Vertex AI model garden docs and adjust `locals.tf`.

- [ ] **Step 4 [DEVELOPER]: Learning-log entry** — cold-start latency of the first request vs a warm request; Ollama vs Gemini answer quality on 5 smoke questions; what IAM/networking error you hit (if any) and how you diagnosed it.

- [ ] **Step 5 [DEVELOPER]: Execute section 6 (destroy) and the checklist**

Expected: every checklist command prints an empty list.

- [ ] **Step 6: Commit**

```bash
git add docs/deploy.md docs/learning-log.md
git commit -m "docs: add deploy runbook and first cloud run-through notes"
```

---

## Sprint 1 done when

All five items of spec §2 are true: compose stack up; both reports ingested; `/ask` and `geoagent ask` return cited answers; smoke set reviewed and logged; one cloud apply → ingest → ask → destroy completed with the checklist clean.
