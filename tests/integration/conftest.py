from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from geoagent.config import Settings
from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations


def recreate_database(settings: Settings, name: str) -> None:
    with connect(
        settings, dbname="postgres", statement_timeout_ms=60_000, register_vector_type=False
    ) as admin:
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Credentials come from the repo `.env` (the same file docker compose reads)."""
    s = Settings()
    if s.db_host not in {"127.0.0.1", "localhost", "::1"}:
        pytest.exit(
            f"refusing to run integration tests against non-local database host {s.db_host!r}",
            returncode=2,
        )
    return s


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
