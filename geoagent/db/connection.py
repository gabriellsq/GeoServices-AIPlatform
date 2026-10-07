import logging
import random
import time
from collections.abc import Callable

import psycopg
from pgvector.psycopg import register_vector

from geoagent.config import Settings

log = logging.getLogger(__name__)


def connect(
    settings: Settings,
    *,
    dbname: str | None = None,
    statement_timeout_ms: int | None = None,
    register_vector_type: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> psycopg.Connection:
    """Open an autocommit connection, retrying transient failures with jittered backoff.

    `settings.db_connect_retries` is the total number of attempts. The conninfo string
    contains the password, so it is never logged. Set `register_vector_type=False` when the
    `vector` extension may not exist yet (migrations, admin connections).
    """
    conninfo = settings.conninfo(dbname=dbname, statement_timeout_ms=statement_timeout_ms)
    attempts = settings.db_connect_retries
    for attempt in range(1, attempts + 1):
        try:
            conn = psycopg.connect(conninfo, autocommit=True)
            break
        except psycopg.OperationalError as exc:
            if attempt == attempts:
                raise
            delay = 0.5 * 2 ** (attempt - 1) + random.uniform(0, 0.25)
            log.warning(
                "database connection failed, retrying",
                extra={
                    "attempt": attempt,
                    "retry_in_s": round(delay, 2),
                    "error": type(exc).__name__,
                },
            )
            sleep(delay)
    if register_vector_type:
        try:
            register_vector(conn)
        except BaseException:
            conn.close()
            raise
    return conn
