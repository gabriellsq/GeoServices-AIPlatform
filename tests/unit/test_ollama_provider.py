import json

import httpx
import pytest

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.ollama import OllamaProvider
from geoagent.providers.types import Generation, LLMProvider


def make(handler, base_url: str = "http://gpu-host:11434") -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(base_url=base_url, model="qwen3:14b", timeout_s=5, client=client)


def ok_body(content: str = "Gold [1].") -> dict:
    return {
        "model": "qwen3:14b",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 120,
        "eval_count": 7,
        "done": True,
    }


def test_sends_chat_request_and_parses_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    gen = make(handler).generate(system="SYS", prompt="USER")

    assert seen["method"] == "POST"
    assert seen["url"] == "http://gpu-host:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "qwen3:14b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert gen == Generation(text="Gold [1].", model="qwen3:14b", tokens_in=120, tokens_out=7)


def test_trailing_slash_in_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=ok_body())

    make(handler, base_url="http://gpu-host:11434/").generate("s", "p")
    assert urls == ["http://gpu-host:11434/api/chat"]


def test_missing_token_counts_default_to_zero():
    body = ok_body()
    del body["prompt_eval_count"], body["eval_count"]
    gen = make(lambda r: httpx.Response(200, json=body)).generate("s", "p")
    assert (gen.tokens_in, gen.tokens_out) == (0, 0)


def test_satisfies_protocol():
    assert isinstance(make(lambda r: httpx.Response(200, json=ok_body())), LLMProvider)


def test_timeout_maps_to_provider_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderTimeout):
        make(handler).generate("s", "p")


def test_connection_refused_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailable):
        make(handler).generate("s", "p")


def test_server_error_maps_to_unavailable():
    with pytest.raises(ProviderUnavailable):
        make(lambda r: httpx.Response(500, json={"error": "boom"})).generate("s", "p")


def test_model_not_pulled_is_a_provider_error_with_message():
    resp = httpx.Response(404, json={"error": "model 'qwen3:14b' not found"})
    with pytest.raises(ProviderError) as info:
        make(lambda r: resp).generate("s", "p")
    assert not isinstance(info.value, ProviderUnavailable)
    assert "not found" in str(info.value)
