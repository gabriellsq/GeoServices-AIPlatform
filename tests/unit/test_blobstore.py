from types import SimpleNamespace

import pytest
from google.api_core.exceptions import NotFound

from geoagent.blobstore.base import BlobNotFound, BlobStore, validate_key
from geoagent.blobstore.gcs import GcsBlobStore
from geoagent.blobstore.local import LocalFsBlobStore


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data, content_type):
        self.bucket.objects[self.name] = (data, content_type)

    def download_as_bytes(self):
        if self.name not in self.bucket.objects:
            raise NotFound("no such object")
        return self.bucket.objects[self.name][0]


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, prefix):
        return [SimpleNamespace(name=n) for n in reversed(list(self.objects)) if n.startswith(prefix)]


def make_gcs():
    bucket = FakeBucket()
    return GcsBlobStore(client=SimpleNamespace(bucket=lambda name: bucket), bucket="my-bucket"), bucket


@pytest.fixture(params=["local", "gcs"])
def store(request, tmp_path) -> BlobStore:
    if request.param == "local":
        return LocalFsBlobStore(root=tmp_path, bucket="raw")
    return make_gcs()[0]


def test_roundtrip_and_overwrite(store):
    store.put("raw/ws/doc.pdf", b"v1", "application/pdf")
    assert store.get("raw/ws/doc.pdf") == b"v1"
    store.put("raw/ws/doc.pdf", b"v2", "application/pdf")
    assert store.get("raw/ws/doc.pdf") == b"v2"


def test_list_keys_sorted_and_prefix_filtered(store):
    store.put("incoming/ws/b.pdf", b"b")
    store.put("incoming/ws/a.pdf", b"a")
    store.put("raw/ws/c.pdf", b"c")
    assert store.list_keys("incoming/ws/") == ["incoming/ws/a.pdf", "incoming/ws/b.pdf"]
    assert store.list_keys("nothing/") == []


def test_missing_key_raises_blob_not_found(store):
    with pytest.raises(BlobNotFound):
        store.get("raw/ws/missing.pdf")


INVALID_KEYS = [
    "", ".", "a/..", "../escape.pdf", "a/../../x", "/etc/x", "a\\b.pdf", "C:\\x",
    "a//b.pdf", "a/./b.pdf", "trailing/", "x.pdf:stream", "nul\x00byte",
]


@pytest.mark.parametrize("key", INVALID_KEYS)
def test_invalid_keys_are_rejected_by_both_stores(store, key):
    with pytest.raises(ValueError):
        store.put(key, b"x")


def test_validate_key_returns_valid_key_unchanged():
    assert validate_key("raw/ws/doc.pdf") == "raw/ws/doc.pdf"


def test_stores_satisfy_protocol(store):
    assert isinstance(store, BlobStore)


def test_uris(tmp_path):
    assert LocalFsBlobStore(tmp_path, "raw").uri_for("raw/ws/doc.pdf") == "local://raw/raw/ws/doc.pdf"
    assert make_gcs()[0].uri_for("raw/ws/x.pdf") == "gs://my-bucket/raw/ws/x.pdf"


def test_local_files_live_under_the_bucket_directory(tmp_path):
    LocalFsBlobStore(tmp_path, "raw").put("raw/ws/doc.pdf", b"x")
    assert (tmp_path / "raw" / "raw" / "ws" / "doc.pdf").read_bytes() == b"x"


def test_local_write_is_atomic(tmp_path, monkeypatch):
    store = LocalFsBlobStore(tmp_path, "raw")
    store.put("raw/doc.pdf", b"old")

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("geoagent.blobstore.local.os.replace", crash)
    with pytest.raises(OSError):
        store.put("raw/doc.pdf", b"new")
    assert store.get("raw/doc.pdf") == b"old"
    assert [p.name for p in (tmp_path / "raw" / "raw").iterdir()] == ["doc.pdf"]


def test_local_list_keys_ignores_temp_files(tmp_path):
    store = LocalFsBlobStore(tmp_path, "raw")
    store.put("incoming/a.pdf", b"a")
    (tmp_path / "raw" / "incoming" / ".a.pdf.0123abcd.tmp").write_bytes(b"partial")
    assert store.list_keys("incoming/") == ["incoming/a.pdf"]


@pytest.mark.parametrize("bucket", ["", ".", "..", "a/b", "a\\b"])
def test_local_rejects_invalid_bucket_names(tmp_path, bucket):
    with pytest.raises(ValueError):
        LocalFsBlobStore(tmp_path, bucket)


def test_gcs_list_keys_skips_folder_placeholders():
    store, bucket = make_gcs()
    bucket.objects["incoming/"] = (b"", None)
    bucket.objects["incoming/a.pdf"] = (b"a", None)
    assert store.list_keys("incoming/") == ["incoming/a.pdf"]
