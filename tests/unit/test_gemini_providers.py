import math
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.gemini import GeminiEmbeddings, VertexGeminiProvider, l2_normalize
from geoagent.providers.types import EmbeddingProvider, Generation, LLMProvider, TaskType


class FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.embed_calls: list = []
        self.generate_calls: list = []

    def _next(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def embed_content(self, *, model, contents, config):
        self.embed_calls.append((model, list(contents), config))
        return self._next()

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append((model, contents, config))
        return self._next()


def emb(*vectors) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=list(v)) for v in vectors])


def gen_response(text="ok", finish="STOP", usage=None, has_candidates=True, block=None):
    candidates = (
        [SimpleNamespace(finish_reason=SimpleNamespace(name=finish) if finish else None)]
        if has_candidates
        else []
    )
    feedback = SimpleNamespace(block_reason=SimpleNamespace(name=block)) if block else None
    return SimpleNamespace(
        text=text, usage_metadata=usage, candidates=candidates, prompt_feedback=feedback
    )


def embedder(models: FakeModels, dim: int = 2, batch_size: int = 2, sleeps: list | None = None):
    record = sleeps if sleeps is not None else []
    return GeminiEmbeddings(
        client=SimpleNamespace(models=models),
        model="gemini-embedding-001",
        dim=dim,
        batch_size=batch_size,
        max_retries=2,
        sleep=record.append,
    )


def test_l2_normalize():
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ProviderError):
        l2_normalize([0.0, 0.0])


def test_embed_normalizes_and_passes_task_type_and_dim():
    models = FakeModels([emb([3, 4])])
    out = embedder(models).embed(["hello"], TaskType.RETRIEVAL_QUERY)
    assert out == [[0.6, 0.8]]
    model, contents, config = models.embed_calls[0]
    assert model == "gemini-embedding-001"
    assert contents == ["hello"]
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == 2


def test_embed_batches_and_preserves_order():
    models = FakeModels([emb([1, 0], [0, 1]), emb([1, 1], [2, 0]), emb([0, 3])])
    out = embedder(models, batch_size=2).embed(list("abcde"), TaskType.RETRIEVAL_DOCUMENT)
    assert [len(call[1]) for call in models.embed_calls] == [2, 2, 1]
    assert len(out) == 5
    assert out[2] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert out[4] == [0.0, 1.0]


def test_empty_input_makes_no_calls():
    models = FakeModels([])
    assert embedder(models).embed([], TaskType.RETRIEVAL_DOCUMENT) == []
    assert models.embed_calls == []


def test_retries_rate_limit_then_succeeds():
    sleeps: list = []
    models = FakeModels([FakeAPIError(429), emb([0, 2])])
    out = embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)
    assert out == [[0.0, 1.0]]
    assert sleeps == [1]


def test_gives_up_after_max_retries():
    models = FakeModels([FakeAPIError(503), FakeAPIError(503), FakeAPIError(503)])
    with pytest.raises(ProviderUnavailable):
        embedder(models).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_client_error_is_not_retried():
    sleeps: list = []
    models = FakeModels([FakeAPIError(400)])
    with pytest.raises(ProviderError) as info:
        embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)
    assert not isinstance(info.value, ProviderUnavailable)
    assert sleeps == []


def test_wrong_dimension_is_rejected():
    models = FakeModels([emb([1, 2, 3])])
    with pytest.raises(ProviderError):
        embedder(models, dim=2).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_embedder_satisfies_protocol():
    assert isinstance(embedder(FakeModels([])), EmbeddingProvider)


def vertex(models: FakeModels) -> VertexGeminiProvider:
    return VertexGeminiProvider(client=SimpleNamespace(models=models), model="gemini-3.8-flash")


def test_vertex_generate_maps_text_and_usage():
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=4)
    models = FakeModels([gen_response(text="Gold [1].", usage=usage)])
    gen = vertex(models).generate(system="SYS", prompt="USER")
    assert gen == Generation(text="Gold [1].", model="gemini-3.8-flash", tokens_in=10, tokens_out=7)
    model, contents, config = models.generate_calls[0]
    assert model == "gemini-3.8-flash"
    assert contents == "USER"
    assert config.system_instruction == "SYS"


def test_vertex_missing_usage_defaults_to_zero():
    models = FakeModels([gen_response(text="ok", finish=None)])
    gen = vertex(models).generate("s", "p")
    assert (gen.text, gen.tokens_in, gen.tokens_out) == ("ok", 0, 0)


def test_vertex_error_mapping():
    req = httpx.Request("POST", "https://example.invalid")
    with pytest.raises(ProviderTimeout):
        vertex(FakeModels([httpx.ReadTimeout("slow", request=req)])).generate("s", "p")
    with pytest.raises(ProviderUnavailable):
        vertex(FakeModels([FakeAPIError(503)])).generate("s", "p")
    with pytest.raises(ProviderError):
        vertex(FakeModels([FakeAPIError(400)])).generate("s", "p")


def test_vertex_satisfies_protocol():
    assert isinstance(vertex(FakeModels([])), LLMProvider)


def test_vertex_no_candidates_is_an_error():
    with pytest.raises(ProviderError, match="no candidates"):
        vertex(FakeModels([gen_response(text=None, has_candidates=False, block="SAFETY")])).generate("s", "p")


@pytest.mark.parametrize("finish", ["SAFETY", "MAX_TOKENS", "RECITATION"])
def test_vertex_incomplete_finish_is_an_error(finish):
    with pytest.raises(ProviderError, match="incomplete"):
        vertex(FakeModels([gen_response(text="partial", finish=finish)])).generate("s", "p")


def test_vertex_empty_answer_is_an_error():
    with pytest.raises(ProviderError, match="empty"):
        vertex(FakeModels([gen_response(text="  ")])).generate("s", "p")


def test_missing_embeddings_is_a_provider_error():
    with pytest.raises(ProviderError):
        embedder(FakeModels([SimpleNamespace(embeddings=None)])).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_transient_transport_error_is_retried():
    sleeps: list = []
    req = httpx.Request("POST", "https://example.invalid")
    models = FakeModels([httpx.ConnectError("reset", request=req), emb([0, 2])])
    assert embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT) == [[0.0, 1.0]]
    assert sleeps == [1]


def test_embedding_timeout_after_retries_is_provider_timeout():
    req = httpx.Request("POST", "https://example.invalid")
    models = FakeModels([httpx.ReadTimeout("slow", request=req) for _ in range(3)])
    with pytest.raises(ProviderTimeout):
        embedder(models).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)


def test_real_sdk_errors_are_classified():
    sleeps: list = []
    rate_limited = genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota"}})
    models = FakeModels([rate_limited, emb([3, 4])])
    assert embedder(models, sleeps=sleeps).embed(["x"], TaskType.RETRIEVAL_DOCUMENT) == [[0.6, 0.8]]
    assert sleeps == [1]
    bad_request = genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})
    with pytest.raises(ProviderError) as info:
        embedder(FakeModels([bad_request])).embed(["x"], TaskType.RETRIEVAL_DOCUMENT)
    assert not isinstance(info.value, ProviderUnavailable)
