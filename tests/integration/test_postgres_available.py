import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_pgvector_extension_is_available():
    with psycopg.connect("postgresql://<DB_USER>:<DB_PASSWORD>@localhost:5432/geoagent") as conn:
        row = conn.execute(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
        ).fetchone()
    assert row is not None, "pgvector is not installed in this Postgres image"
