# GeoAgent

Retrieval-augmented generation over public geoscience technical reports (NI 43-101), built as a
production-style GenAI learning project: plain Python, Postgres + pgvector, Gemini embeddings,
Ollama locally, Vertex AI + Cloud Run + Cloud SQL in the cloud, Terraform for infrastructure.

- Design: `docs/superpowers/specs/2026-10-06-geoagent-mvp-design.md`
- Plan: `docs/superpowers/plans/2026-10-06-geoagent-mvp.md`
- Learning log: `docs/learning-log.md`

Source reports are copyrighted and are not stored in this repository; `scripts/fetch_reports.py`
downloads them into the git-ignored `data/` directory.

## Local setup

1. Copy `.env.example` to `.env` (git-ignored) and fill in `DB_USER` and `DB_PASSWORD`. Generate a
   password with `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
2. Start the database: `docker compose up -d postgres`.

Postgres only reads the credentials when the `pgdata` volume is first created. If you change
them later, run `docker compose down -v` before starting again (this deletes the local database).
