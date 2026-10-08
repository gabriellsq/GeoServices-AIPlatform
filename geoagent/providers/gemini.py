import math
import time
from collections.abc import Callable
from typing import Any

import httpx
from google.genai import types

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import Generation, TaskType

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ProviderError("embedding has zero norm")
    return [x / norm for x in vector]


class GeminiEmbeddings:
    """gemini-embedding-001 via google-genai (AI Studio key locally, Vertex AI in cloud).

    Vectors smaller than 3072 dims are NOT normalised by the API, so we normalise here.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        dim: int,
        batch_size: int = 32,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sleep = sleep

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size], task_type))
        return vectors

    def _embed_batch(self, batch: list[str], task_type: TaskType) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type.value, output_dimensionality=self.dim
        )
        for attempt in range(self.max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(f"embedding request timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise ProviderUnavailable(f"embedding service unreachable: {exc}") from exc
            except Exception as exc:
                code = _status_code(exc)
                if code in RETRYABLE_CODES and attempt < self.max_retries:
                    self.sleep(min(2**attempt, 30))
                    continue
                if code in RETRYABLE_CODES:
                    raise ProviderUnavailable(f"embedding failed after retries: {exc}") from exc
                raise ProviderError(f"embedding failed: {exc}") from exc
            values = [list(e.values) for e in result.embeddings]
            if len(values) != len(batch):
                raise ProviderError(f"expected {len(batch)} embeddings, got {len(values)}")
            for v in values:
                if len(v) != self.dim:
                    raise ProviderError(f"expected dim {self.dim}, got {len(v)}")
            return [l2_normalize(v) for v in values]
        raise AssertionError("unreachable")


class VertexGeminiProvider:
    """Gemini on Vertex AI. Temperature is left at the model default (Gemini 3 guidance)."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, system: str, prompt: str) -> Generation:
        config = types.GenerateContentConfig(system_instruction=system)
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Gemini request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"Gemini unreachable: {exc}") from exc
        except Exception as exc:
            if _status_code(exc) in RETRYABLE_CODES:
                raise ProviderUnavailable(f"Gemini unavailable: {exc}") from exc
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        usage = resp.usage_metadata
        tokens_in = (getattr(usage, "prompt_token_count", None) or 0) if usage else 0
        tokens_out = (
            (getattr(usage, "candidates_token_count", None) or 0)
            + (getattr(usage, "thoughts_token_count", None) or 0)
            if usage
            else 0
        )
        return Generation(
            text=resp.text or "", model=self.model, tokens_in=tokens_in, tokens_out=tokens_out
        )
