import re
from typing import Protocol, runtime_checkable

MAX_KEY_BYTES = 1024
_FORBIDDEN_CHARS = re.compile(r"[\\:\x00-\x1f\x7f]")


class BlobNotFound(LookupError):
    """No object exists under this key."""


def validate_key(key: str) -> str:
    """Reject keys that would mean different things in different stores.

    A key is '/'-separated, non-empty segments: no backslashes, colons or control characters,
    no '.' or '..' segments, no leading/trailing '/'. Returns the key unchanged.
    """
    if not key or len(key.encode("utf-8")) > MAX_KEY_BYTES:
        raise ValueError("blob key must be 1-1024 bytes")
    if _FORBIDDEN_CHARS.search(key):
        raise ValueError(f"blob key contains a forbidden character: {key!r}")
    if any(segment in ("", ".", "..") for segment in key.split("/")):
        raise ValueError(f"blob key has an empty, '.' or '..' segment: {key!r}")
    return key


@runtime_checkable
class BlobStore(Protocol):
    """Opaque object storage. Keys look like 'raw/<workspace>/<document_id>.pdf'.

    Keys are validated with validate_key; get raises BlobNotFound for missing objects.
    """

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes under key and return the object's URI."""
        ...

    def get(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> list[str]:
        """Return keys starting with prefix, sorted."""
        ...

    def uri_for(self, key: str) -> str: ...
