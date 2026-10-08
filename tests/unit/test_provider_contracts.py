import dataclasses

import pytest

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import EmbeddingProvider, Generation, LLMProvider, TaskType
from tests.fakes import FakeEmbedder, FakeLLM


def test_task_types_are_gemini_strings():
    assert TaskType.RETRIEVAL_DOCUMENT == "RETRIEVAL_DOCUMENT"
    assert TaskType.RETRIEVAL_QUERY == "RETRIEVAL_QUERY"


def test_generation_is_frozen_dataclass():
    g = Generation(text="t", model="m", tokens_in=1, tokens_out=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.text = "changed"  # type: ignore[misc]


def test_fakes_satisfy_protocols():
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmbedder(), EmbeddingProvider)


def test_object_without_generate_is_not_an_llm_provider():
    class NotAProvider:
        model = "m"

    assert not isinstance(NotAProvider(), LLMProvider)


def test_error_hierarchy():
    assert issubclass(ProviderUnavailable, ProviderError)
    assert issubclass(ProviderTimeout, ProviderError)
    assert not issubclass(ProviderTimeout, ProviderUnavailable)
