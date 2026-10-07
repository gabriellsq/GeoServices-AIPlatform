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
