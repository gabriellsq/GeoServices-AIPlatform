import hashlib

import pytest

from geoagent.blobstore.local import LocalFsBlobStore
from geoagent.db.workspaces import create_workspace
from geoagent.ingest.chunker import chunk_pages
from geoagent.ingest.parse import parse_pdf
from geoagent.ingest.pipeline import ingest_bytes, ingest_from_blob
from geoagent.providers.errors import ProviderUnavailable
from geoagent.providers.types import TaskType
from tests.fakes import FakeEmbedder
from tests.fixtures.fixture_pdf import build_fixture_pdf

pytestmark = pytest.mark.integration


@pytest.fixture
def blobs(tmp_path):
    return LocalFsBlobStore(root=tmp_path, bucket="raw")


@pytest.fixture
def ws(conn):
    return create_workspace(conn, slug="test-ws", name="Test")


@pytest.fixture
def data():
    return build_fixture_pdf("a")


def run(conn, blobs, embedder, ws, data, **kw):
    return ingest_bytes(conn, blobs, embedder, workspace_id=ws, data=data, title="Report", **kw)


def chunk_rows(conn, doc_id):
    return conn.execute(
        "SELECT ordinal, kind, workspace_id, page_start, page_end, section, label_start, "
        "label_end, text, token_count FROM chunks WHERE document_id = %s ORDER BY ordinal",
        (doc_id,),
    ).fetchall()


def test_stores_blob_document_and_chunks(conn, blobs, ws, data):
    result = run(conn, blobs, FakeEmbedder(), ws, data)
    assert (result.status, result.skipped, result.error) == ("ready", False, None)
    key = f"raw/{ws}/{result.document_id}.pdf"
    assert blobs.get(key) == data
    row = conn.execute(
        "SELECT status, error, page_count, title, sha256, blob_uri FROM documents WHERE id = %s",
        (result.document_id,),
    ).fetchone()
    assert row == ("ready", None, 4, "Report", hashlib.sha256(data).hexdigest(), blobs.uri_for(key))
    expected = chunk_pages(parse_pdf(data))
    assert len(chunk_rows(conn, result.document_id)) == len(expected)


def test_chunks_embedded_as_documents(conn, blobs, ws, data):
    embedder = FakeEmbedder()
    run(conn, blobs, embedder, ws, data)
    assert len(embedder.calls) == 1
    assert embedder.calls[0][1] == TaskType.RETRIEVAL_DOCUMENT


def test_chunk_rows_carry_metadata(conn, blobs, ws, data):
    result = run(conn, blobs, FakeEmbedder(), ws, data)
    rows = chunk_rows(conn, result.document_id)
    assert all(r[2] == ws and r[3] <= r[4] and r[9] > 0 for r in rows)
    assert "7.1 Regional Geology" in {r[5] for r in rows}
    assert {"table", "text"} <= {r[1] for r in rows}
    assert all(r[6] is not None for r in rows)


def test_stored_text_matches_chunker_output(conn, blobs, ws, data):
    result = run(conn, blobs, FakeEmbedder(), ws, data)
    expected = [(c.ordinal, c.text) for c in chunk_pages(parse_pdf(data))]
    assert [(r[0], r[8]) for r in chunk_rows(conn, result.document_id)] == expected


def test_reingest_is_skipped(conn, blobs, ws, data):
    first = run(conn, blobs, FakeEmbedder(), ws, data)
    embedder = FakeEmbedder()
    second = run(conn, blobs, embedder, ws, data)
    assert second.skipped and second.document_id == first.document_id
    assert embedder.calls == []
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (1,)


def test_same_bytes_in_other_workspace_is_separate(conn, blobs, ws, data):
    other = create_workspace(conn, slug="other-ws", name="Other")
    a = run(conn, blobs, FakeEmbedder(), ws, data)
    b = run(conn, blobs, FakeEmbedder(), other, data)
    assert a.document_id != b.document_id and not b.skipped
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (2,)


def test_embedding_failure_marks_document_failed(conn, blobs, ws, data):
    embedder = FakeEmbedder(fail_with=ProviderUnavailable("quota exhausted"))
    result = run(conn, blobs, embedder, ws, data)
    assert result.status == "failed" and "quota exhausted" in result.error
    row = conn.execute(
        "SELECT status, error FROM documents WHERE id = %s", (result.document_id,)
    ).fetchone()
    assert row[0] == "failed" and "quota exhausted" in row[1]
    assert chunk_rows(conn, result.document_id) == []


def test_retry_after_failure_reuses_document(conn, blobs, ws, data):
    failed = run(conn, blobs, FakeEmbedder(fail_with=ProviderUnavailable("down")), ws, data)
    retried = run(conn, blobs, FakeEmbedder(), ws, data)
    assert retried.status == "ready" and not retried.skipped
    assert retried.document_id == failed.document_id
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (1,)
    assert conn.execute("SELECT error FROM documents").fetchone() == (None,)
    assert chunk_rows(conn, retried.document_id)


def test_invalid_pdf_fails_without_raising(conn, blobs, ws):
    result = run(conn, blobs, FakeEmbedder(), ws, b"nope")
    assert result.status == "failed" and result.error


def test_wrong_vector_count_fails(conn, blobs, ws, data):
    class ShortEmbedder(FakeEmbedder):
        def embed(self, texts, task_type):
            return super().embed(texts, task_type)[:-1]

    result = run(conn, blobs, ShortEmbedder(), ws, data)
    assert result.status == "failed"
    assert chunk_rows(conn, result.document_id) == []


def test_ingest_from_blob(conn, blobs, ws):
    blobs.put(f"incoming/{ws}/alpha.pdf", build_fixture_pdf("a"), "application/pdf")
    blobs.put(f"incoming/{ws}/beta.PDF", build_fixture_pdf("b"), "application/pdf")
    blobs.put(f"incoming/{ws}/notes.txt", b"hello", "text/plain")
    results = ingest_from_blob(
        conn, blobs, FakeEmbedder(), workspace_id=ws, prefix=f"incoming/{ws}/"
    )
    assert [r.title for r in results] == ["alpha", "beta"]
    assert all(r.status == "ready" for r in results)
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (2,)
