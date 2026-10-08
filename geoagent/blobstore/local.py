import os
import uuid
from pathlib import Path

from geoagent.blobstore.base import BlobNotFound, validate_key

_TMP_SUFFIX = ".tmp"


class LocalFsBlobStore:
    """Filesystem stand-in for a cloud bucket: <root>/<bucket>/<key>.

    Writes are atomic (temp file + os.replace), like a single-request GCS upload.
    `content_type` is accepted for interface parity and ignored.
    """

    def __init__(self, root: Path, bucket: str) -> None:
        if not bucket or bucket in (".", "..") or "/" in bucket or "\\" in bucket:
            raise ValueError(f"invalid bucket name: {bucket!r}")
        self.bucket = bucket
        self.base = (Path(root) / bucket).resolve()

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = (self.base / key).resolve()
        if path == self.base or not path.is_relative_to(self.base):
            raise ValueError(f"key escapes the bucket: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise BlobNotFound(key)
        return path.read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        if not self.base.exists():
            return []
        keys = (
            p.relative_to(self.base).as_posix()
            for p in self.base.rglob("*")
            if p.is_file() and not (p.name.startswith(".") and p.name.endswith(_TMP_SUFFIX))
        )
        return sorted(k for k in keys if k.startswith(prefix))

    def uri_for(self, key: str) -> str:
        return f"local://{self.bucket}/{validate_key(key)}"
