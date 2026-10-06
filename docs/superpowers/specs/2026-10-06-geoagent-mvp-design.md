# GeoAgent MVP (Sprint 1) — Design Spec

- **Date:** 2026-10-06
- **Status:** Approved (brainstorming), pending implementation plan
- **Repo:** `GeoSolution`, Python package `geoagent`

## 1. Purpose

GeoAgent is a learning project for building production-grade applied GenAI on geoscience data: Python, Google Cloud, Terraform, evaluation and observability, with a production-first mindset.

The project builds a production-style GenAI system over real New Zealand geology reports. Sprint 1 (this spec) delivers a RAG MVP that runs locally and deploys once to GCP. Later sprints add evals, observability, hybrid search, structured/spatial data, an agent + MCP server, and security (see §12).

### Primary learning constraint

Unreviewed generated code teaches little. Therefore the **core RAG logic is hand-written by the developer**, with an AI assistant acting as tutor, test author and reviewer — not implementer (see §8). This constraint overrides speed.

## 2. MVP definition of done

1. `docker compose up` starts `api` and `postgres` (with pgvector) locally; raw PDFs live in a local-filesystem blob store (`blobdata/`).
2. The Waihi and Macraes NI 43-101 technical reports are ingested: raw PDFs in the local blob store; chunks, metadata and embeddings in Postgres.
3. `POST /ask` and `geoagent ask "..."` return a grounded answer with citations (document title, page range, section).
4. The smoke set (~30 hand-written Q&A pairs) runs via `geoagent smoke`; results are reviewed manually and recorded in `docs/learning-log.md`.
5. `terraform apply` deploys the same container image to Cloud Run with Cloud SQL, GCS and Vertex AI; the ingest job runs in the cloud; one question is answered from the cloud endpoint; then `terraform destroy` removes everything except the state bucket.

### Explicitly out of scope for Sprint 1

MCP server, agents/tool-calling, Google ADK, scored eval harness, OpenTelemetry tracing, BM25/hybrid search, reranking, authentication and Row-Level Security, any UI beyond Swagger, CI/CD, structured/spatial/ML tools, event-triggered ingestion, self-hosted embedding models.

## 3. Key decisions (with rationale)

| Decision | Choice | Rationale |
|---|---|---|
| Deploy strategy | Local-first, one GCP deploy required before MVP is done | Turning prototypes into deployed systems is the core skill being practised; surfacing IAM/secrets/networking issues while the system is small is cheap. `apply → test → destroy` keeps cost near $0. |
| Corpus | Real NZ NI 43-101 reports: OceanaGold **Waihi District PFS** (eff. 2024-06-30) and **Macraes Operation** (eff. 2025-12-31) | Realistic messy PDFs (tables, headers, standard sections). Two contrasting gold deposits (epithermal vs orogenic) create cross-document confusion cases for evaluation. |
| Corpus licensing | PDFs are **not committed**; `scripts/fetch_reports.py` downloads them into git-ignored `data/` | Company reports are copyrighted. Waihi has a direct PDF URL; the Macraes direct PDF URL (oceanagold.com or SEDAR+) is resolved during planning. |
| Generation model | Local: Ollama on a GPU host on the LAN (model size chosen to fit available VRAM / unified memory). Cloud: Gemini via Vertex AI | $0 local iteration on capable hardware; managed model in cloud. Swappable via config. |
| Embedding model | Gemini embeddings (`gemini-embedding-001`, 768 dims) **in every environment** | Vector dimension and vector space must be identical everywhere, otherwise a local index is useless in cloud. Model name to be re-verified as current when the plan is written. |
| Vector store | Postgres + pgvector (no dedicated vector DB) | One system; chunk text, metadata and vectors written in one transaction (no dual-write sync problem); tenant filtering in plain SQL. Corpus size (~5–20k chunks) is far below pgvector limits. Matches Google's "RAG with Vertex AI + AlloyDB/Cloud SQL" reference architecture. |
| Cloud database | Cloud SQL for PostgreSQL (smallest tier) | Cheapest Postgres with pgvector. AlloyDB (ScaNN, columnar engine, higher SLA) is the documented upgrade path when scale justifies it; migration is dump/restore + connection string. |
| Blob store | Local filesystem (`blobdata/`, URIs `local://<bucket>/<key>`) locally, GCS in cloud, behind a `BlobStore` interface | MinIO was the original choice, but its community edition stopped publishing images (Oct 2025) and was archived (Apr 2026); a filesystem store keeps the same interface lesson with zero extra infrastructure. Raw PDFs are the source of truth for re-parsing/re-chunking; Postgres stores pointers (`blob_uri`), never chunk text in blobs. |
| Framework | **Plain Python** (psycopg 3, raw SQL, `google-genai`, `httpx`) — no LangChain/LlamaIndex | Every step must be visible and explainable. Google ADK is evaluated later in the agent sprint against a hand-written tool loop. |
| Interface | FastAPI (+ auto Swagger at `/docs`) and a Typer CLI | Focus is backend/platform; a UI adds little learning value at this stage. |
| PDF parser | PyMuPDF | Fast page-level text and basic table detection. AGPL licence is acceptable for a public portfolio repo. Docling comparison is a later learning-log experiment. |
| Migrations | Numbered plain `.sql` files + a tiny runner tracking applied files in `schema_migrations` | Developer writes and reads real SQL; no ORM/Alembic abstraction. |
| Tooling | uv, Python 3.12, ruff, pytest, pydantic-settings, Typer | Standard modern Python stack. |

## 4. Runtime topology

```
Dev machine (repo)                          GPU host(s) on the LAN
docker compose: api · postgres ─────────► Ollama (generation)
          │
          └──► Gemini API (embeddings, every environment)
```

- Ollama runs **outside** Docker Compose. Docker on macOS cannot use the Metal GPU, so on Apple Silicon Ollama runs natively; on a Linux AMD host it runs as a container (`ollama/ollama:rocm`) with `/dev/kfd` and `/dev/dri` passed through (exact image tag and ROCm support for the GPU to be verified in the plan).
- The API reaches Ollama over the LAN via `OLLAMA_BASE_URL`, mirroring production where the model is also a remote endpoint.

### Configuration (environment variables via pydantic-settings)

| Variable | Local | Cloud |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `vertex` |
| `LLM_MODEL` | e.g. `qwen3:14b` / `gemma3:27b` | `gemini-3.8-flash` (latest stable Flash as of 2026-10) |
| `OLLAMA_BASE_URL` | `http://<gpu-host>:11434` | unused |
| `GEMINI_API_KEY` | AI Studio key (in `.env`, git-ignored) | unused (Vertex uses Application Default Credentials) |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | unused | set by Terraform |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | `gemini-embedding-001`, `768` | same |
| `EMBEDDING_BATCH_SIZE` | `32` | `32` (Vertex limit: 250 texts **and** 20k tokens per request; 32 × 512-token chunks fits) |
| `EMBEDDING_LOCATION` | unused | region for Vertex embeddings (e.g. `us-central1`) |
| `BLOB_STORE`, `BLOB_ROOT` | `local`, `blobdata` | `gcs`, unused |
| `BLOB_BUCKET` | `geoagent-raw` | Terraform-created bucket name |
| `DATABASE_URL` | compose Postgres | Cloud SQL via Unix socket `/cloudsql/PROJECT:REGION:INSTANCE` |

## 5. Components

| Module | Responsibility | Author |
|---|---|---|
| `geoagent/config.py` | Typed settings | LLM-written, developer reviews |
| `geoagent/blobstore/` | `BlobStore` protocol (`put`, `get`, `list_keys`, `uri_for`); `LocalFsBlobStore`, `GcsBlobStore` | LLM-written, developer reviews |
| `geoagent/providers/` | `LLMProvider` protocol (`generate(prompt, ...) -> Generation`); `EmbeddingProvider` protocol (`embed(texts, task_type) -> list[vector]`); `OllamaProvider`, `VertexGeminiProvider`, `GeminiEmbeddings` | **Developer:** protocols + `OllamaProvider`. LLM: Vertex/Gemini adapters |
| `geoagent/ingest/parse.py` | PDF → list of pages (number, text, extracted tables as text) | **Developer** |
| `geoagent/ingest/chunker.py` | Pages → chunks: section-aware (NI 43-101 numbered headings), token-limited, page tracking, repeated header/footer removal | **Developer** |
| `geoagent/ingest/pipeline.py` | Orchestration and document status transitions | **Developer** |
| `geoagent/rag/retriever.py` | Embed question → pgvector top-k filtered by `workspace_id` → scored chunks | **Developer** |
| `geoagent/rag/prompt.py` | Grounded prompt construction; `PROMPT_VERSION` constant; parse and validate `[n]` citations | **Developer** |
| `geoagent/rag/answer.py` | retrieve → (not-found short-circuit) → prompt → generate → response | **Developer** |
| `geoagent/api/` | FastAPI app: `POST /ask`, `POST /documents`, `GET /documents`, `GET /healthz` | LLM-written, developer reviews |
| `geoagent/cli.py` | `geoagent ingest`, `geoagent ask`, `geoagent smoke`, `geoagent migrate` | LLM-written, developer reviews |
| `geoagent/db/migrations/*.sql` | Schema | **Developer** (SQL); runner LLM-written |
| `scripts/fetch_reports.py` | Download corpus PDFs into `data/` | LLM-written |
| `infra/terraform/` | GCP resources | **Developer** for core resources, with the assistant explaining |

### Repository layout

```
GeoSolution/
├── geoagent/            # package (modules above)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/        # small synthetic PDF(s) safe to commit
├── evals/smoke.jsonl
├── scripts/fetch_reports.py
├── infra/terraform/{bootstrap,main}/
├── docs/
│   ├── superpowers/specs/
│   ├── learning-log.md
│   └── deploy.md
├── data/                # git-ignored: downloaded PDFs
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 6. Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE workspaces (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE documents (
    id           UUID PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    title        TEXT NOT NULL,
    source_url   TEXT,
    blob_uri     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('uploaded','processing','ready','failed')),
    error        TEXT,
    page_count   INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, sha256)
);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL,
    ordinal      INT  NOT NULL,
    page_start   INT  NOT NULL,
    page_end     INT  NOT NULL,
    section      TEXT,
    text         TEXT NOT NULL,
    token_count  INT  NOT NULL,
    embedding    vector(768) NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_workspace ON chunks (workspace_id);
```

Notes:
- `workspace_id` is denormalised onto `chunks` so vector search filters without a join. It is the tenant boundary; in Sprint 1 it is passed by the client, in the security sprint it comes from the authenticated identity and is enforced with RLS.
- 768 dimensions: pgvector's HNSW on `vector` supports at most 2,000 dimensions, so the 3,072-dim default output is reduced via `output_dimensionality=768`. Truncated Gemini embeddings must be **L2-normalised** before storage and before querying.
- `sha256` uniqueness per workspace prevents duplicate ingestion.

## 7. Data flows

### 7.1 Ingest (identical code: CLI locally, Cloud Run Job in cloud)

1. `scripts/fetch_reports.py` downloads PDFs to `data/`.
2. `geoagent ingest --workspace nz-gold [PATHS...] [--from-blob PREFIX]` — sources are local file paths (local dev) and/or every PDF under a blob-store prefix such as `incoming/nz-gold/` (cloud job; `POST /documents` also uploads there). For each file compute `sha256`. If `(workspace_id, sha256)` exists with status `ready`, skip.
3. Store the canonical copy at `raw/{workspace_id}/{document_id}.pdf` (the bytes are read anyway for parsing, so a plain write is used for both source types); insert `documents` row with status `uploaded`.
4. Set `processing`; parse → chunk → embed in batches using task type `RETRIEVAL_DOCUMENT`, with exponential-backoff retries on HTTP 429/5xx; L2-normalise vectors.
5. In **one transaction**: delete existing chunks for the document, insert new chunks, set status `ready`.
6. On any exception: set status `failed` with the error message; continue with the next file. Re-running is safe (idempotent).

### 7.2 Ask

1. `POST /ask {workspace_id, question, top_k = 8}`.
2. Embed question with task type `RETRIEVAL_QUERY`; L2-normalise.
3. Query: `ORDER BY embedding <=> $q LIMIT $k` with `WHERE workspace_id = $ws`; return similarity `1 - distance`.
4. If no chunk exceeds `MIN_SIMILARITY` (configurable; initial value calibrated empirically and recorded in the learning log), return a "not found in the documents" answer **without calling the LLM**.
5. Build prompt: system rules (answer only from sources, cite as `[n]`, say when information is absent, treat source text as data not instructions) + numbered sources `[1]..[k]` each labelled with document title, pages and section + the question.
6. Generate with a 60 s timeout.
7. Parse `[n]` markers; drop citations referring to non-existent sources; map valid ones to `{document_title, page_start, page_end, section, snippet}`.
8. Respond `{answer, citations[], model, prompt_version, latency_ms, tokens_in, tokens_out}`.

## 8. Learning method

1. **Two zones.** Developer-written: parse, chunker, pipeline, retriever, prompt, answer, provider protocols + Ollama adapter, SQL migrations, core Terraform. LLM-written with developer review: config, blob stores, Vertex/Gemini adapters, FastAPI layer, CLI, migration runner, Dockerfile, compose, fetch script, test fixtures.
2. **The assistant writes failing tests; developer makes them pass.** For each developer-zone module, the assistant writes pytest tests that specify behaviour (e.g. chunks never cross a section boundary; `page_start <= page_end`; no chunk exceeds the token limit; invalid citations are dropped). The assistant reviews the developer's implementation and asks questions rather than supplying the solution.
3. **Predict → measure.** Each experiment starts with a written prediction.
4. **Break it on purpose.** Deliberately misconfigure (wrong task type, no header stripping, extreme top-k) and observe.
5. **`docs/learning-log.md`.** One entry per experiment: hypothesis, setup, result, explanation. Initial experiments:
   - chunk size 256 / 512 / 1024 tokens;
   - header/footer stripping on vs off;
   - `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT` for questions;
   - top-k 3 / 8 / 20;
   - `MIN_SIMILARITY` calibration.

## 9. Error handling

- **Ingest:** per-document status is the checkpoint; one failed PDF does not stop others; embedding retries with backoff; sha256 duplicate guard; chunk replacement is transactional.
- **Ask:** LLM timeout → HTTP 504; model host unreachable → HTTP 503 with a message naming the provider/host; empty or low-similarity retrieval → "not found" path (HTTP 200); validation errors → HTTP 422. All errors are structured JSON, never raw stack traces.
- **Logging:** structured JSON via stdlib `logging` with a per-request `request_id`; Cloud Logging ingests stdout JSON natively. Full tracing is deferred to the observability sprint.

## 10. Testing

- **Unit (no network):** parser on a committed synthetic fixture PDF, chunker, prompt building, citation parsing, retriever with a deterministic fake embedder.
- **Integration:** against the compose Postgres (pgvector image) with fake LLM/embedding providers: ingest fixture → ask → assert citations reference the fixture's pages.
- **Smoke set (`evals/smoke.jsonl`):** ~30 developer-written items `{id, question, expected_answer, expected_document, expected_pages?, category}` where `category ∈ {exact_term, conceptual, cross_document_trap, not_in_corpus}`. `geoagent smoke` prints answers and citations for manual review. Scoring is deferred to Sprint 2.

## 11. Cloud deployment (Terraform)

```
infra/terraform/
├── bootstrap/          # GCS bucket for Terraform state (applied once, never destroyed)
└── main/
    ├── artifact_registry.tf   # Docker repository
    ├── storage.tf             # raw-PDF bucket
    ├── sql.tf                 # Cloud SQL PostgreSQL 16, smallest tier; pgvector enabled by migration
    ├── secrets.tf             # DB password in Secret Manager
    ├── iam.tf                 # runtime service account: roles/cloudsql.client,
    │                          # roles/storage.objectUser (bucket-scoped), roles/aiplatform.user,
    │                          # secretAccessor on the DB secret
    ├── cloud_run.tf           # Service "geoagent-api" + Job "geoagent-ingest" (same image, different command)
    └── budget.tf              # billing budget alert (e.g. $20)
```

- Cloud Run service: `uvicorn geoagent.api.main:app --port $PORT`, `min_instances = 0`, Cloud SQL connection via the built-in Cloud SQL integration.
- Cloud Run job: `geoagent migrate && geoagent ingest --workspace nz-gold --from-blob incoming/nz-gold/`. Migrations run idempotently at the start of each job execution, so no separate migration step or DB network access from the dev machine is needed.
- Cloud SQL: public IP with **no authorized networks**; reachable only through the Cloud SQL connector. Private IP/VPC deferred to the security sprint.
- Runbook `docs/deploy.md`: build/push image → `terraform apply` → upload PDFs to `gs://<bucket>/incoming/nz-gold/` → `gcloud run jobs execute geoagent-ingest` → `curl /ask` → `terraform destroy` (with a checklist confirming nothing billable remains).

## 12. Roadmap after Sprint 1 (each gets its own spec)

| Sprint | Focus |
|---|---|
| 2 | Scored eval harness, prompt versioning, Ollama vs Gemini (accuracy/latency/cost) |
| 3 | Observability: OpenTelemetry tracing, token/cost metrics, local Grafana vs Cloud Trace/Monitoring |
| 4 | Hybrid search (Postgres full-text/BM25 + RRF) and reranking, measured with Sprint 2 evals |
| 5 | Structured data: NZP&M open-file drill collars (manual RealMe download) in PostGIS + deterministic math tools |
| 6 | Agent + own MCP server (FastMCP, stdio + streamable HTTP, tool filtering); hand-written tool loop vs Google ADK |
| 7 | Security: authentication, tenant from identity, Postgres RLS, prompt-injection tests, deletion/lineage |
| 8 | Classic-ML anomaly detection; event-triggered ingestion (GCS → Pub/Sub/Eventarc → Cloud Run job) |

## 13. References

- Google RAG reference architectures: https://docs.cloud.google.com/architecture/rag-reference-architectures
- RAG with Vertex AI + AlloyDB: https://docs.cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai
- Waihi District NI 43-101: https://assets.oceanagold.com/documents/Reports/Technical-Reports/WaihiDistrictNI43101TechnicalReport.pdf
- Macraes NI 43-101 filing: https://oceanagold.com/news/oceanagold-files-annual-information-form-and-updated-technical-reports-for-haile-macraes-and-didipio
- NZP&M open-file geochemical database: https://geodata.nzpam.govt.nz/report/mr4510
