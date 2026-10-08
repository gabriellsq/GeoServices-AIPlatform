import uuid
from collections.abc import Iterator

import psycopg
import pytest

from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations
from tests.integration.conftest import recreate_database

pytestmark = pytest.mark.integration

MIGRATION_DB = "geoagent_migration_test"


@pytest.fixture
def fresh(settings) -> Iterator[psycopg.Connection]:
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


def test_embedding_dimension_is_enforced(fresh):
    apply_migrations(fresh)
    ws = make_workspace(fresh)
    doc = make_document(fresh, ws)
    with pytest.raises(psycopg.errors.DataException):
        fresh.execute(
            "INSERT INTO chunks (id, document_id, workspace_id, ordinal, page_start, page_end, "
            "text, token_count, embedding) "
            "VALUES (%s, %s, %s, 0, 1, 1, 'x', 1, array_fill(0.1, ARRAY[767])::vector)",
            (uuid.uuid4(), doc, ws),
        )


@pytest.mark.parametrize(("ordinal", "page_start", "page_end"), [(-1, 1, 1), (0, 0, 1), (0, 3, 2)])
def test_chunk_position_checks(fresh, ordinal, page_start, page_end):
    apply_migrations(fresh)
    ws = make_workspace(fresh)
    doc = make_document(fresh, ws)
    with pytest.raises(psycopg.errors.CheckViolation):
        fresh.execute(
            "INSERT INTO chunks (id, document_id, workspace_id, ordinal, page_start, page_end, "
            "text, token_count, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, 'x', 1, array_fill(0.1, ARRAY[768])::vector)",
            (uuid.uuid4(), doc, ws, ordinal, page_start, page_end),
        )
