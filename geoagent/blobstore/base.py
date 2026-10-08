from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Opaque object storage. Keys look like 'raw/<workspace>/<document_id>.pdf'."""

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes under key and return the object's URI."""
        ...

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> list[str]:
        """Return keys starting with prefix, sorted."""
        ...

    def uri_for(self, key: str) -> str: ...
