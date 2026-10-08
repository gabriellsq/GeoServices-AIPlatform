import uuid

import psycopg
import pytest

from geoagent.db.workspaces import WorkspaceNotFound, create_workspace, get_workspace_id

pytestmark = pytest.mark.integration


def test_create_then_resolve_by_slug(conn):
    workspace_id = create_workspace(conn, slug="tenant-a", name="Tenant A")
    assert isinstance(workspace_id, uuid.UUID)
    assert get_workspace_id(conn, "tenant-a") == workspace_id


def test_unknown_slug_raises(conn):
    with pytest.raises(WorkspaceNotFound):
        get_workspace_id(conn, "nobody")


def test_duplicate_slug_is_rejected(conn):
    create_workspace(conn, slug="tenant-a", name="Tenant A")
    with pytest.raises(psycopg.errors.UniqueViolation):
        create_workspace(conn, slug="tenant-a", name="Again")
