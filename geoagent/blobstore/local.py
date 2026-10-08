from pathlib import Path


class LocalFsBlobStore:
    """Filesystem stand-in for a cloud bucket: <root>/<bucket>/<key>."""

    def __init__(self, root: Path, bucket: str) -> None:
        self.bucket = bucket
        self.base = (Path(root) / bucket).resolve()

    def _path(self, key: str) -> Path:
        path = (self.base / key).resolve()
        if not path.is_relative_to(self.base):
            raise ValueError(f"key escapes the bucket: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        if not self.base.exists():
            return []
        keys = (p.relative_to(self.base).as_posix() for p in self.base.rglob("*") if p.is_file())
        return sorted(k for k in keys if k.startswith(prefix))

    def uri_for(self, key: str) -> str:
        return f"local://{self.bucket}/{key}"
