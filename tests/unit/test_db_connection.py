import logging

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from geoagent.config import Settings
from geoagent.db import connection

PASSWORD = "hunter2-not-real"


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, db_user="app", db_password=PASSWORD, **overrides)


def test_retries_with_backoff_then_succeeds(monkeypatch):
    calls, sleeps, sentinel = [], [], object()

    def fake_connect(conninfo, autocommit):
        calls.append(conninfo)
        if len(calls) < 3:
            raise psycopg.OperationalError("connection refused")
        return sentinel

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    conn = connection.connect(
        settings(db_connect_retries=3), register_vector_type=False, sleep=sleeps.append
    )
    assert conn is sentinel
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] < 0.75 and 1.0 <= sleeps[1] < 1.25


def test_gives_up_after_configured_attempts(monkeypatch):
    attempts = []

    def fake_connect(conninfo, autocommit):
        attempts.append(1)
        raise psycopg.OperationalError("down")

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    with pytest.raises(psycopg.OperationalError):
        connection.connect(
            settings(db_connect_retries=2), register_vector_type=False, sleep=lambda s: None
        )
    assert len(attempts) == 2


def test_passes_dbname_statement_timeout_and_autocommit(monkeypatch):
    seen = {}

    def fake_connect(conninfo, autocommit):
        seen.update(conninfo=conninfo, autocommit=autocommit)
        return object()

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    connection.connect(
        settings(), dbname="other", statement_timeout_ms=120000, register_vector_type=False
    )
    params = conninfo_to_dict(seen["conninfo"])
    assert params["dbname"] == "other"
    assert params["options"] == "-c statement_timeout=120000"
    assert seen["autocommit"] is True


def test_retry_logs_never_contain_the_password(monkeypatch, caplog):
    def fake_connect(conninfo, autocommit):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(connection.psycopg, "connect", fake_connect)
    with caplog.at_level(logging.DEBUG), pytest.raises(psycopg.OperationalError):
        connection.connect(settings(), register_vector_type=False, sleep=lambda s: None)
    assert caplog.records, "retries should be logged"
    assert PASSWORD not in caplog.text


def test_closes_connection_if_vector_registration_fails(monkeypatch):
    class FakeConn:
        closed = False

        def close(self):
            self.closed = True

    fake = FakeConn()
    monkeypatch.setattr(connection.psycopg, "connect", lambda conninfo, autocommit: fake)

    def boom(conn):
        raise psycopg.ProgrammingError("vector type not found")

    monkeypatch.setattr(connection, "register_vector", boom)
    with pytest.raises(psycopg.ProgrammingError):
        connection.connect(settings())
    assert fake.closed
