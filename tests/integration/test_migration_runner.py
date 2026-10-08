import threading

import psycopg
import pytest

from geoagent.db.connection import connect
from geoagent.db.migrate import apply_migrations
from tests.integration.conftest import recreate_database

pytestmark = pytest.mark.integration

RUNNER_DB = "geoagent_runner_test"


@pytest.fixture
def runner_db(settings) -> str:
    recreate_database(settings, RUNNER_DB)
    return RUNNER_DB


def open_runner_conn(settings) -> psycopg.Connection:
    return connect(settings, dbname=RUNNER_DB, register_vector_type=False)


def public_tables(conn) -> set[str]:
    rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    return {r[0] for r in rows}


def test_failure_in_any_file_rolls_back_the_whole_run(settings, runner_db, tmp_path):
    (tmp_path / "001_ok.sql").write_text("CREATE TABLE ok_table (id int);", encoding="utf-8")
    (tmp_path / "002_bad.sql").write_text(
        "CREATE TABLE bad_table (id int); SELECT 1/0;", encoding="utf-8"
    )
    with open_runner_conn(settings) as conn:
        with pytest.raises(psycopg.errors.DivisionByZero):
            apply_migrations(conn, tmp_path)
        assert public_tables(conn).isdisjoint({"ok_table", "bad_table", "schema_migrations"})


def test_file_with_bom_is_applied(settings, runner_db, tmp_path):
    (tmp_path / "001_bom.sql").write_text("CREATE TABLE bom_table (id int);", encoding="utf-8-sig")
    with open_runner_conn(settings) as conn:
        assert apply_migrations(conn, tmp_path) == ["001_bom.sql"]


def test_requires_autocommit_connection(settings, runner_db, tmp_path):
    with psycopg.connect(settings.conninfo(dbname=RUNNER_DB)) as conn:
        with pytest.raises(ValueError):
            apply_migrations(conn, tmp_path)


def test_concurrent_runners_apply_each_file_once(settings, runner_db):
    results, errors = [], []

    def run():
        try:
            with open_runner_conn(settings) as conn:
                results.append(apply_migrations(conn))
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(results, key=len) == [[], [], [], ["001_init.sql"]]
