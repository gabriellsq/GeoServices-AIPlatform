# Learning Notes — GeoAgent

Cheat sheet of the design decisions in this project and the reasoning behind them.
Each section ends with how the same problem is solved on Google Cloud.

---

## 1. Why Postgres + pgvector instead of a dedicated vector database

### What pgvector is
- A Postgres **extension**: adds a `vector(N)` column type, distance operators and ANN indexes.
- Embeddings are just a column next to the chunk text and metadata.
- Operators: `<=>` cosine distance · `<->` L2 · `<#>` negative inner product. Similarity = `1 - (a <=> b)`.
- Without an index → exact scan (fine for tens of thousands of rows). With **HNSW** → approximate nearest neighbour (ms over millions of rows).
- The index operator class must match the query operator (`vector_cosine_ops` ↔ `<=>`), otherwise the index is silently ignored.

### pgvector does NOT compute embeddings
It stores and searches vectors. The vectors come from a model (Gemini `gemini-embedding-001` here), called from Python.

Ways to embed *inside* the database (exist, not used here):

| Option | How |
|---|---|
| `google_ml_integration` (Cloud SQL / AlloyDB) | `google_ml.embedding(model, text)` calls Vertex AI from SQL |
| AlloyDB AI | Same, plus managed/automatic embedding generation |
| `pgai`, `pg_vectorize` | Extensions that call model APIs and keep embeddings in sync with tables |

**Why embed in the application instead:** batching, retries with backoff, task types (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`), normalisation, observability and swapping providers are all easier in code. Model calls inside the database put slow, failure-prone network calls inside transactions.
**When in-DB embedding wins:** keeping embeddings automatically fresh when rows change (trigger-style), small teams, SQL-first workflows.

### Dedicated vector DB vs Postgres

| | Postgres + pgvector | Dedicated (Qdrant, Pinecone, Weaviate, Vertex Vector Search) |
|---|---|---|
| Systems to run | 1 | 2 (operational DB + vector store) |
| Consistency | Chunk + vector + metadata in **one transaction** | Dual writes → you must build sync |
| Tenant filtering / joins | Plain SQL `WHERE`, RLS | Metadata filters, more limited |
| Backups, access control | Existing Postgres tooling | Separate |
| Scale ceiling | ~10–100M vectors per node | Billions, sharded |
| Extras | Fewer knobs | Quantization, hybrid search, multi-tenancy built in |

**The real cost of two stores = the sync problem.** Example: a confidential report is deleted in the DB, the vector delete fails → retrieval still serves its text.
How it is solved when you do have two stores:
1. **Transactional outbox** — domain change + an `outbox` event row in the same DB transaction; a worker applies it to the vector store (at-least-once → deletes must be idempotent; deterministic chunk IDs).
2. **CDC** — read the Postgres WAL (Debezium → Kafka → sink).
3. **Read-time check** — vector hits are re-validated against the source of truth (still exists? still permitted?).
4. **Reconciliation job** — periodic ID diff between stores repairs drift.

**Rule of thumb:** start with pgvector; move to a dedicated store when vector count, query rate or latency targets outgrow one Postgres node — and only then pay for the sync machinery.

### Dimensions and Matryoshka truncation
- `gemini-embedding-001` outputs up to 3072 dims; we store **768**.
- **Matryoshka Representation Learning (MRL):** the model is *trained* so the first *k* dims are a usable embedding on their own (nested dolls). Information is concentrated at the front, so truncating keeps most retrieval quality.
- Information-theory view: truncation is **lossy compression by dropping the least informative dimensions** — like keeping the top components in PCA. Plain (non-MRL) embeddings spread information evenly, so naive truncation hurts much more.
- Two different compression axes — they combine:

  | Technique | Reduces | Example |
  |---|---|---|
  | MRL truncation | number of dimensions | 3072 → 768 |
  | Quantization | bits per dimension | float32 → `halfvec` (16-bit) → int8 → `binary_quantize` (1-bit) |

- **Truncated vectors must be re-normalised** (L2) — only the full 3072-dim output is unit-length.
- pgvector limits: HNSW on `vector` ≤ 2,000 dims; on `halfvec` ≤ 4,000 dims.
- pgvector has the building blocks in SQL (since 0.7): `subvector(v, start, count)`, `l2_normalize(v)`, `binary_quantize(v)`. Pattern **"shortlist then rerank"**: index a short prefix, fetch top-20 with it, re-order those by the full vector.

  ```sql
  CREATE INDEX ON items USING hnsw ((subvector(embedding, 1, 768)::vector(768)) vector_cosine_ops);
  SELECT * FROM (
      SELECT * FROM items
      ORDER BY subvector(embedding, 1, 768)::vector(768) <=> subvector($q, 1, 768)
      LIMIT 20
  ) s ORDER BY embedding <=> $q LIMIT 5;
  ```
- In this project: truncation happens at the API (`output_dimensionality=768`), normalisation in Python (`GeminiEmbeddings`).
- Reading: Hugging Face blog "Introduction to Matryoshka Embedding Models"; Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022 (arXiv:2205.13147).

### Raw files vs searchable data
- **Blob store** (GCS / local folder) holds the original PDF — source of truth for re-parsing and re-chunking.
- **Postgres** holds metadata, a pointer (`blob_uri`), chunk text and vectors.
- Chunk text lives in the DB (it goes straight into the prompt); the blob is opaque until processed.

### GCP design — RAG storage

```
GCS bucket (raw PDFs) ──event──► Pub/Sub ──► Cloud Run job (parse → chunk → embed)
                                                   │ Vertex AI embeddings
                                                   ▼
                              Cloud SQL / AlloyDB for PostgreSQL + pgvector
                                                   ▲
User ─► Load balancer ─► Cloud Run service (API) ──┘ ─► Vertex AI Gemini (generation)
```
- Google's reference architectures: *RAG with Vertex AI + AlloyDB* (vectors next to operational data — closest to this project), *RAG with Vertex AI Vector Search* (very large scale), *RAG with GKE + Cloud SQL* (maximum control).
- **Cloud SQL** for a PoC (cheapest). **AlloyDB** when scale/latency/analytics justify it: ScaNN index, columnar engine (HTAP), higher SLA, read pools. Both speak Postgres → migration is dump/restore + a connection string.
- **Vertex AI Vector Search** when vectors reach hundreds of millions+ or need very high QPS — accept the sync problem.

---

## 2. Multi-tenancy

### Vocabulary
- **Tenant** = one customer organisation. **Multi-tenant** = one deployment serves many tenants.
- **Tenant filter** = `WHERE tenant_id = X` on every query.
- **Noisy neighbour** = one tenant's heavy usage degrades the others.

### Isolation models

| Model | How | Good for | Cost |
|---|---|---|---|
| **Pool** | Shared tables + `workspace_id` column | Many small tenants | Isolation depends on the filter always being applied |
| **Bridge** | Schema / namespace / partition per tenant | Tens–thousands of tenants, easy offboarding | Per-namespace limits, migrations × N |
| **Silo** | Database / index / project per tenant | Regulated or enterprise tenants, per-tenant keys | Cost and operations grow linearly |

This project: **pool**, with `workspaces` as the tenant table.

### Enforcement — defence in depth
1. **Tenant ID comes from the authenticated identity** (token), never from the request body and never from the LLM.
2. **Repository layer** adds the tenant filter to every query; application code never queries tables directly.
3. **Database enforces it too:** Postgres **Row-Level Security** — the DB refuses other tenants' rows even if a `WHERE` is forgotten.
4. **Schema makes inconsistency impossible:** composite foreign key `chunks(document_id, workspace_id) → documents(id, workspace_id)` — a chunk can never carry a different tenant than its document.
5. **Isolation tests in CI:** "tenant A asks with tenant B's ID → gets nothing".

### Schema decisions
- **Primary keys should be meaningless and immutable.** `workspaces.id` is a UUID; the human-readable `slug` is a separate unique column.
  A slug as PK leaks tenant names into URLs/logs, is guessable, and a rename touches every referencing row and blob path.
- **Denormalising `workspace_id` onto `chunks`** makes the vector query filter without a join; the composite FK removes the risk of the two copies disagreeing.
- **No `ON DELETE CASCADE` from workspaces:** deleting a tenant must be an explicit, audited offboarding process.

### Vector search + tenant filter gotcha
- HNSW finds the nearest *k* across **all** tenants, then the filter removes the others → you may get 2 results instead of 8.
- Fixes: pgvector ≥ 0.8 **iterative index scans** (`SET hnsw.iterative_scan = strict_order`), partial indexes per large tenant, partitioning by tenant, or a bridge/silo model.

### Noisy neighbours
- `statement_timeout` (5 s for API queries here) cancels runaway queries before they starve other tenants.
- Later: per-tenant rate limits and quotas.

### Vector databases — the same three models

| Model | Mechanism |
|---|---|
| Pool | Qdrant payload index (`is_tenant`), Milvus partition key, Vertex Vector Search **restricts** |
| Bridge | Pinecone namespaces, Weaviate native multi-tenancy (one shard per tenant) |
| Silo | Collection / index per tenant |

### GCP design — multi-tenancy

| Model | GCP building blocks |
|---|---|
| Pool | Cloud SQL/AlloyDB + pgvector + **RLS**; or **Vertex AI Vector Search restricts** (each datapoint tagged with a tenant token in a namespace; queries pass an allow-list); Spanner tables keyed by tenant |
| Bridge | Schema or database per tenant on one AlloyDB cluster; Firestore collection path per tenant |
| Silo | **Project per tenant** (from the same Terraform modules) → separate IAM, billing, quotas; per-tenant **CMEK** keys so a tenant can revoke access |

Cross-cutting: tenant identity from **Identity Platform / IAP** tokens, **VPC Service Controls** perimeter, audit logs per tenant.

---

## 3. Multiple users and conversation memory

### Users inside a tenant
- **Authentication** = who you are. **Authorization** = what you may do. Different layers.
- Roles per workspace (e.g. owner / editor / viewer); search returns only data the user is allowed to see.
- Memory and history are per **user**, not just per tenant — colleagues do not see each other's conversations unless designed in.

### Two kinds of memory

| | Short-term (session history) | Long-term (learned facts) |
|---|---|---|
| What | Last N turns of this conversation | "prefers g/t", "works on Macraes" |
| Key | `(tenant_id, user_id, session_id)` | `(tenant_id, user_id)` |
| Store | Document store with TTL; cache for hot sessions | Vector store under the user's namespace, retrieved like RAG |
| Context management | Sliding window + rolling summary within a token budget | Extract after the session, dedupe, retrieve top few |

### Risks
- **Cross-user leakage** — memory keyed only by tenant.
- **Memory poisoning** — a prompt injection saved into long-term memory returns in every future session. Validate writes; store memory as data, never as instructions.
- **Deletion / right to be forgotten** — data spreads to sessions, memory, caches, logs, traces, eval sets. Needs lineage + TTLs, same as document deletion.
- **Cost** — history grows every prompt; summarise or truncate.
- **The model itself does not remember** RAG content — retrieval only fills the prompt for one request. Fine-tuning on customer data *does* embed it in weights (only fix: retrain) → prefer RAG for customer data.

### GCP design — users and memory

```
User ─► Identity Platform / IAP (token: tenant_id, user_id, roles)
     ─► Cloud Run API (authz check; tenant + user taken from the token)
          ├─ Firestore: sessions/{session}/messages (TTL policy)       ← short-term
          ├─ Memorystore (Redis): hot session cache
          ├─ AlloyDB/pgvector or Vector Search (per-user namespace)    ← long-term
          └─ Vertex AI Gemini
     Managed alternative: Vertex AI Agent Engine — Sessions + Memory Bank (integrates with ADK)
```

---

## 4. Secrets and configuration

- **No credentials in git, ever** — not even dev placeholders. Real values live only in a git-ignored `.env`; `.env.example` ships empty fields.
- If a secret is committed: rotate it, and scrub **all** history (`git filter-repo --replace-text`, then `git reflog expire --expire=now --all && git gc --prune=now`) **before** pushing. After a push, assume it is leaked.
- Settings (`pydantic-settings`):
  - credentials **required, no defaults** → app refuses to start without them;
  - `SecretStr` → hidden in `repr`, `str`, JSON;
  - `hide_input_in_errors=True` → validation errors do not echo raw input (otherwise the password lands in logs);
  - `env_ignore_empty=True` → `DB_PASSWORD=` counts as missing;
  - bounds on numeric settings (`connect_timeout=0` means *wait forever* in libpq);
  - `frozen=True` → shared cached settings cannot be mutated.
- `SecretStr` does not protect derived strings: the libpq `conninfo()` contains the password — never log it.
- Terraform: `.tf` and `.terraform.lock.hcl` go in git; **state never does** (it stores secrets in plain text) → remote backend (GCS bucket: versioning, locking, restricted IAM).
- GCP: password generated by Terraform → Secret Manager → injected into Cloud Run at runtime; better still, **Cloud SQL IAM database authentication** (no password exists). CI authenticates with **Workload Identity Federation** (no JSON keys).

---

## 5. Timeouts and failure handling

- **Budgets nest:** each inner timeout is shorter than its caller. Here: proxy/request 90 s > LLM 60 s > DB query 5 s ≥ DB connect 5 s.
- **Interactive requests fail fast; batch jobs wait.** API: 3 connect attempts with jittered backoff, then **503 + `Retry-After`**. Ingest job: longer statement timeout (120 s), job-level retries.
- Timeouts cap the **worst-case cost** (Cloud Run bills while a request is open).
- **Never auto-retry LLM generation** — the provider may still bill the abandoned call.
- `connect_timeout` turned a 261 s hang into a 5 s failure. Root cause was environmental: on Windows, WSL's `wslrelay` holds `[::1]:5432`, so `localhost` → IPv6 → black hole. Use `127.0.0.1`.
- Dev Postgres binds to `127.0.0.1` only — never expose a dev database to the LAN.
- Healthchecks should test what clients use (TCP `-h 127.0.0.1`), not a side channel (unix socket during init).

---

## 6. Schema migrations

- **Concurrent runners race.** Two deploys/jobs running migrations at once → `CREATE TABLE IF NOT EXISTS` / `CREATE EXTENSION` collide (reproduced: 4 runners → 3 crashed).
  Fix: take `pg_advisory_xact_lock(<constant>)` **before** touching `schema_migrations`, then re-read what is applied under the lock. Waiters block, then see "nothing to do".
- **All-or-nothing:** run every pending file in one transaction — Postgres DDL is transactional, so a failure in file N rolls back files 1..N-1 and the bookkeeping rows.
- Prefer the **transaction-scoped** lock (`_xact_`): released automatically on commit/rollback; a session lock on an autocommit connection leaks if you forget to unlock.
- Exceptions to "everything in a transaction": `CREATE INDEX CONCURRENTLY`, some `ALTER TYPE ... ADD VALUE` — they need their own non-transactional step.
- Read SQL files as `utf-8-sig` (Windows editors add a BOM → `syntax error at or near "\ufeff"`).
- Pin extension images (`pgvector/pgvector:0.8.7-pg16`, not `:pg16`) so behaviour you rely on (iterative index scans need ≥ 0.8) cannot silently change.
- Test databases: refuse to run destructive test setup (`DROP DATABASE`) unless the host is local.

---

## 7. Embeddings as data (data-engineering view) + ACID

- **HNSW** = Hierarchical Navigable Small World: a layered graph; search enters at a sparse top layer and descends toward the nearest neighbours. Approximate, fast, memory-hungry. It is an *index*, not a storage design.
- **Embeddings are derived data**, not source data:
  ```
  BRONZE raw PDFs (blob)  →  SILVER pages/chunks (text + metadata)  →  GOLD embeddings + index (rebuildable)
  ```
- Store **full fidelity once**, derive cheap forms (768-dim truncation, binary quantization) like views.
- Record **lineage on every vector**: `embedding_model`, `model_version`, `dims`, `task_type`, `normalized`. Vectors from different models live in different spaces — never mix them in one index.
- **Model change = backfill + index swap** (blue/green): re-embed into a new column/table → build index → switch reads → drop old. Never half-update in place.
- **Serving ≠ analytics:** Postgres/pgvector for low-latency per-tenant top-k (OLTP); export embeddings to **BigQuery/Parquet** for clustering, dedup, drift and topic analysis (`VECTOR_SEARCH`, `ML.GENERATE_EMBEDDING`) — that is the "embedding data mart".
- Truncated vector in Postgres — options:

  | Option | Verdict |
  |---|---|
  | Expression index on `subvector(embedding_full,1,768)` + rerank with full vector | Best: always consistent, no refresh |
  | Generated stored column + index | Consistent, costs storage (functions must be immutable) |
  | Materialized view + index | Stale until `REFRESH`; fine for analytics snapshots, wrong for serving |
  | Plain view | Cannot be ANN-indexed |

- This MVP stores only 768 dims (API truncates) → simpler, but going back to 3072 needs re-embedding. Experiment idea: `halfvec(3072)` + 768-dim expression index, compare recall.
- **HNSW operations:** build the index *after* bulk loads (much faster; raise `maintenance_work_mem`); tune `m`/`ef_construction` (build) and `hnsw.ef_search` (query recall vs latency); updates/deletes leave dead graph entries until `VACUUM` → bloat and recall loss → periodic `REINDEX` under churn.

### ACID still matters
| | In this project |
|---|---|
| Atomicity | Replace chunks + set `ready` in one transaction — never `ready` with half the chunks |
| Consistency | Composite FK, `vector(768)`, CHECKs always enforced — a chunk can never point at another tenant's document |
| Isolation | MVCC: a query during re-ingestion sees all old or all new chunks |
| Durability | HNSW index is WAL-logged; survives crashes with the rows |

- ACID guarantees the **state** is correct; **ANN results are still approximate**. Consistency ≠ exact nearest neighbours — it means never returning a deleted or foreign chunk.
- Dedicated vector DBs often trade this away: write-to-searchable freshness lag, tunable consistency levels (e.g. Strong/Bounded/Session/Eventually), no transactions spanning your metadata store.
- ACID matters most for **deletions, permissions, re-ingestion, model swaps**; eventual consistency is acceptable for an append-only public corpus.

---

## 8. Choosing the store — scale vs control

| Optimises for | Dedicated vector DBs | Postgres + pgvector |
|---|---|---|
| Goal | Distributed ANN: sharding, replication, QPS, quantization | Integrity: transactions, constraints, joins, SQL |
| Consistency | Often eventual/tunable (write → searchable lag) | Strong after commit |
| Integrity rules | Application-level | In the engine (FK, CHECK, UNIQUE, RLS) |
| Ad-hoc questions | Limited filter APIs | Any SQL |

- Not a law that "distributed = less control": **Spanner** and **AlloyDB** are distributed *and* ACID with vector search. The real trade is that **control at scale costs money and complexity**.

### Why store chunks next to vectors at all?
Nobody reads the 768 numbers — but the rows around them are queried all the time:
- **Every request:** chunk text + title + pages → prompt and citations.
- **Debugging:** "what did retrieval return for this question, with what scores?"
- **Operations:** failed documents and why, chunk counts per document.
- **Compliance:** delete a tenant, find everything embedded with model v1.
- **Quality:** near-duplicate chunks, similarity distributions, norm checks, drift — vectors are inspected *statistically*.

Vector DBs store payloads too (Qdrant JSON, Pinecone metadata, Weaviate objects). The deciding question is **what else lives around the vectors**: tenants, users, permissions, statuses, jobs, audit → if you need those with transactions and joins, a separate vector DB becomes a *second* store (dual-write sync problem). Document NoSQL (Firestore, MongoDB Atlas) = flexible schema + scale + vector search, but no joins, limited transactions, integrity in code.

### When Postgres / Cloud SQL is the right choice
1. Relational operational data already exists; vectors are one feature of it.
2. Deletes and permission changes must apply immediately (regulated data, strict tenant isolation).
3. **Hybrid queries in one statement:** vector similarity + relational filters + full-text (`tsvector`) + **PostGIS** — e.g. "chunks about porphyry copper, from drill reports within 5 km of this point, for tenant A". Strong argument in geoscience, where nearly everything has a location.
4. Up to tens of millions of vectors at moderate QPS per node.
5. Cost (one small instance vs always-on vector nodes) and team skills (SQL, backups, migrations).

### When something else
| Need | GCP choice |
|---|---|
| 100M–billions of vectors, very high QPS, sharded ANN | Vertex AI Vector Search (accept eventual consistency + sync) |
| Global scale + strong consistency + vectors | Spanner |
| Bigger Postgres, HTAP, higher SLA | AlloyDB (ScaNN, columnar engine) |
| Document-shaped app data, real-time client sync | Firestore + vector search |
| Batch analytics over embeddings | BigQuery `VECTOR_SEARCH` |

**Rule:** match the data's centre of gravity — mostly relational with vectors as a feature → Postgres; mostly vectors at massive scale → vector DB; documents synced to clients → Firestore; analytics → BigQuery.
