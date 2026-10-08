from typing import Any

from google.api_core.exceptions import NotFound

from geoagent.blobstore.base import BlobNotFound, validate_key


class GcsBlobStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self.bucket_name = bucket
        self._bucket = client.bucket(bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._bucket.blob(validate_key(key)).upload_from_string(data, content_type=content_type)
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        try:
            return self._bucket.blob(validate_key(key)).download_as_bytes()
        except NotFound as exc:
            raise BlobNotFound(key) from exc

    def list_keys(self, prefix: str) -> list[str]:
        # Objects ending in "/" are console "folder" placeholders, not data.
        return sorted(b.name for b in self._bucket.list_blobs(prefix=prefix) if not b.name.endswith("/"))

    def uri_for(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{validate_key(key)}"
