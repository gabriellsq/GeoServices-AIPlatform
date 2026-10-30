"""Ingestion pipeline: PDF bytes -> blob + document row -> chunks -> embeddings -> Postgres.

Stages, tracked in `documents.status`: uploaded (row + blob written) -> processing (parse,
chunk, embed) -> ready, or failed with the error text. Ingest is idempotent per
(workspace, sha256): a `ready` document is skipped, a `failed` one is retried under the same id.

Transaction rule: the embedding call is a network call and never runs inside a transaction.
Chunks are replaced and the document marked ready in ONE short transaction, and failures are
recorded on the autocommit connection after that transaction has rolled back.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
import psycopg

from geoagent.blobstore.base import BlobStore
from geoagent.ingest.chunker import chunk_pages
from geoagent.ingest.parse import parse_pdf
from geoagent.providers.types import EmbeddingProvider, TaskType

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    title: str
    status: str  # "ready" or "failed"
    skipped: bool
    error: str | None = None


def _set_status(conn: psycopg.Connection, document_id: uuid.UUID, status: str, error=None) -> None:
    conn.execute(
        "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
        (status, error, document_id),
    )


def ingest_bytes(
    conn: psycopg.Connection,
    blobs: BlobStore,
    embedder: EmbeddingProvider,
    *,
    workspace_id: uuid.UUID,
    data: bytes,
    title: str,
    source_url: str | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> IngestResult:
    """Ingest one PDF into an existing workspace. Never raises for processing errors.

    The document row is upserted first, so every failure after that point is recorded on it.
    A database error during the upsert itself has no row to record on and propagates.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    # The id is only used on insert; on conflict the existing id and blob_uri are kept, which
    # is what makes retries reuse the document and concurrent runners converge on one row.
    candidate = uuid.uuid4()
    row = conn.execute(
        "INSERT INTO documents (id, workspace_id, title, source_url, blob_uri, sha256, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, 'uploaded') "
        "ON CONFLICT (workspace_id, sha256) DO UPDATE SET status = 'uploaded', error = NULL, "
        "title = EXCLUDED.title, source_url = EXCLUDED.source_url, updated_at = now() "
        "WHERE documents.status <> 'ready' RETURNING id",
        (
            candidate,
            workspace_id,
            title,
            source_url,
            blobs.uri_for(_blob_key(workspace_id, candidate)),
            sha256,
        ),
    ).fetchone()
    if row is None:  # conflict on a ready document: the WHERE clause left it untouched
        existing = conn.execute(
            "SELECT id, title FROM documents WHERE workspace_id = %s AND sha256 = %s",
            (workspace_id, sha256),
        ).fetchone()
        return IngestResult(existing[0], existing[1], "ready", skipped=True)
    document_id = row[0]

    try:
        blobs.put(_blob_key(workspace_id, document_id), data, "application/pdf")
        _set_status(conn, document_id, "processing")
        pages = parse_pdf(data)
        chunks = chunk_pages(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts, TaskType.RETRIEVAL_DOCUMENT) if texts else []
        if len(vectors) != len(chunks):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (id, document_id, workspace_id, ordinal, kind, page_start, "
                    "page_end, label_start, label_end, section, text, token_count, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            uuid.uuid4(),
                            document_id,
                            workspace_id,
                            c.ordinal,
                            c.kind,
                            c.page_start,
                            c.page_end,
                            c.label_start,
                            c.label_end,
                            c.section,
                            c.text,
                            c.token_count,
                            np.array(v, dtype=np.float32),
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
            conn.execute(
                "UPDATE documents SET status = 'ready', page_count = %s, error = NULL, "
                "updated_at = now() WHERE id = %s",
                (len(pages), document_id),
            )
    except Exception as exc:
        # Recorded outside the chunk transaction, so it survives that rollback.
        log.warning(
            "ingestion failed",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
        _set_status(conn, document_id, "failed", str(exc))
        return IngestResult(document_id, title, "failed", skipped=False, error=str(exc))
    return IngestResult(document_id, title, "ready", skipped=False)


def _blob_key(workspace_id: uuid.UUID, document_id: uuid.UUID) -> str:
    return f"raw/{workspace_id}/{document_id}.pdf"


def ingest_from_blob(
    conn: psycopg.Connection,
    blobs: BlobStore,
    embedder: EmbeddingProvider,
    *,
    workspace_id: uuid.UUID,
    prefix: str,
) -> list[IngestResult]:
    """Ingest every `.pdf` under `prefix` in the blob store (case-insensitive extension)."""
    return [
        ingest_bytes(
            conn,
            blobs,
            embedder,
            workspace_id=workspace_id,
            data=blobs.get(key),
            title=PurePosixPath(key).stem,
            source_url=blobs.uri_for(key),
        )
        for key in blobs.list_keys(prefix)
        if key.lower().endswith(".pdf")
    ]
