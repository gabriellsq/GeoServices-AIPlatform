import uuid

import psycopg


class WorkspaceNotFound(LookupError):
    """No workspace has this slug."""


def create_workspace(conn: psycopg.Connection, *, slug: str, name: str) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO workspaces (id, slug, name) VALUES (%s, %s, %s)",
        (workspace_id, slug, name),
    )
    return workspace_id


def get_workspace_id(conn: psycopg.Connection, slug: str) -> uuid.UUID:
    row = conn.execute("SELECT id FROM workspaces WHERE slug = %s", (slug,)).fetchone()
    if row is None:
        raise WorkspaceNotFound(slug)
    return row[0]
