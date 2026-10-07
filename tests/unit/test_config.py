import pytest
from psycopg.conninfo import conninfo_to_dict
from pydantic import ValidationError

from geoagent.config import Settings


@pytest.fixture
def env(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("DB_USER", "app_user")
    monkeypatch.setenv("DB_PASSWORD", "s3cret-value")
    return monkeypatch


def make() -> Settings:
    return Settings(_env_file=None)


def test_defaults_target_local_development(env):
    s = make()
    assert (s.db_host, s.db_port, s.db_name) == ("127.0.0.1", 5432, "geoagent")
    assert (s.llm_provider, s.llm_model) == ("ollama", "qwen3:8b")
    assert s.ollama_base_url == "http://127.0.0.1:11434"
    assert s.blob_store == "local"
    assert (s.embedding_model, s.embedding_dim, s.embedding_batch_size) == (
        "gemini-embedding-001", 768, 32,
    )
    assert s.top_k == 8
    assert (s.db_connect_timeout_s, s.db_connect_retries, s.db_statement_timeout_ms) == (5, 3, 5000)
    assert s.llm_timeout_s == 60.0


@pytest.mark.parametrize("missing", ["DB_USER", "DB_PASSWORD"])
def test_db_credentials_are_required(env, missing):
    env.delenv(missing)
    with pytest.raises(ValidationError) as info:
        make()
    assert missing.lower() in str(info.value)


def test_secrets_are_never_rendered(env):
    env.setenv("GEMINI_API_KEY", "AIza-not-a-real-key")
    s = make()
    for rendered in (repr(s), str(s), s.model_dump_json()):
        assert "s3cret-value" not in rendered
        assert "AIza-not-a-real-key" not in rendered
    assert s.db_password.get_secret_value() == "s3cret-value"
    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == "AIza-not-a-real-key"


def test_conninfo_builds_libpq_parameters(env):
    assert conninfo_to_dict(make().conninfo()) == {
        "host": "127.0.0.1",
        "port": "5432",
        "dbname": "geoagent",
        "user": "app_user",
        "password": "s3cret-value",
        "connect_timeout": "5",
        "options": "-c statement_timeout=5000",
    }


def test_conninfo_overrides(env):
    params = conninfo_to_dict(make().conninfo(dbname="geoagent_test", statement_timeout_ms=120000))
    assert params["dbname"] == "geoagent_test"
    assert params["options"] == "-c statement_timeout=120000"


def test_environment_overrides(env):
    env.setenv("LLM_PROVIDER", "vertex")
    env.setenv("BLOB_STORE", "gcs")
    env.setenv("TOP_K", "3")
    env.setenv("DB_HOST", "/cloudsql/proj:region:inst")
    s = make()
    assert (s.llm_provider, s.blob_store, s.top_k) == ("vertex", "gcs", 3)
    assert conninfo_to_dict(s.conninfo())["host"] == "/cloudsql/proj:region:inst"


def test_validation_errors_do_not_echo_secrets(env):
    env.delenv("DB_USER")
    env.setenv("GEMINI_API_KEY", "AIza-not-a-real-key")
    with pytest.raises(ValidationError) as info:
        make()
    assert "s3cret-value" not in str(info.value)
    assert "AIza-not-a-real-key" not in str(info.value)


def test_empty_values_are_treated_as_missing(env):
    env.setenv("GEMINI_API_KEY", "")
    assert make().gemini_api_key is None
    env.setenv("DB_PASSWORD", "")
    with pytest.raises(ValidationError):
        make()


@pytest.mark.parametrize(
    ("var", "value"),
    [("DB_PORT", "99999"), ("DB_CONNECT_TIMEOUT_S", "0"), ("DB_CONNECT_RETRIES", "0"),
     ("DB_STATEMENT_TIMEOUT_MS", "0")],
)
def test_out_of_range_values_are_rejected(env, var, value):
    env.setenv(var, value)
    with pytest.raises(ValidationError):
        make()


def test_settings_are_immutable(env):
    s = make()
    with pytest.raises(ValidationError):
        s.top_k = 50
