from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATION_LOCK_KEY = 0x67656F61  # arbitrary constant; serialises concurrent runners


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending *.sql files in filename order, all-or-nothing.

    Runs in one transaction under a transaction-scoped advisory lock: concurrent runners wait,
    then see the files as applied. Requires an autocommit connection.
    """
    if not conn.autocommit:
        raise ValueError("apply_migrations needs an autocommit connection")
    applied: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            conn.execute(path.read_text(encoding="utf-8-sig"))  # tolerate a BOM
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied
