import psycopg
import pytest

from geoagent.config import Settings

pytestmark = pytest.mark.integration


def test_pgvector_extension_is_available():
    with psycopg.connect(Settings().conninfo()) as conn:
        row = conn.execute(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
        ).fetchone()
    assert row is not None, "pgvector is not installed in this Postgres image"
