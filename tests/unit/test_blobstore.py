from types import SimpleNamespace

from geoagent.blobstore.base import BlobStore
from geoagent.blobstore.gcs import GcsBlobStore
from geoagent.blobstore.local import LocalFsBlobStore


def test_local_put_get_roundtrip(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    uri = store.put("raw/ws/doc.pdf", b"%PDF-1.7 data", "application/pdf")
    assert uri == "local://raw/raw/ws/doc.pdf"
    assert store.get("raw/ws/doc.pdf") == b"%PDF-1.7 data"
    assert (tmp_path / "raw" / "raw" / "ws" / "doc.pdf").exists()


def test_local_list_keys_is_sorted_and_prefix_filtered(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    store.put("incoming/ws/b.pdf", b"b")
    store.put("incoming/ws/a.pdf", b"a")
    store.put("raw/ws/c.pdf", b"c")
    assert store.list_keys("incoming/ws/") == ["incoming/ws/a.pdf", "incoming/ws/b.pdf"]
    assert store.list_keys("nothing/") == []


def test_local_rejects_path_traversal(tmp_path):
    store = LocalFsBlobStore(root=tmp_path, bucket="raw")
    try:
        store.put("../escape.pdf", b"x")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a key escaping the bucket")


def test_local_store_satisfies_protocol(tmp_path):
    assert isinstance(LocalFsBlobStore(root=tmp_path, bucket="raw"), BlobStore)


def test_gcs_uri_and_put_use_bucket():
    uploaded = {}

    class FakeBlob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, data, content_type):
            uploaded[self.name] = (data, content_type)

    fake_bucket = SimpleNamespace(blob=FakeBlob)
    fake_client = SimpleNamespace(bucket=lambda name: fake_bucket)
    store = GcsBlobStore(client=fake_client, bucket="my-bucket")
    assert store.put("raw/ws/x.pdf", b"x", "application/pdf") == "gs://my-bucket/raw/ws/x.pdf"
    assert uploaded["raw/ws/x.pdf"] == (b"x", "application/pdf")
    assert isinstance(store, BlobStore)
