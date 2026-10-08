from typing import Any


class GcsBlobStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self.bucket_name = bucket
        self._bucket = client.bucket(bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._bucket.blob(key).upload_from_string(data, content_type=content_type)
        return self.uri_for(key)

    def get(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(blob.name for blob in self._bucket.list_blobs(prefix=prefix))

    def uri_for(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{key}"
