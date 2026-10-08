"""Ollama LLM provider using the /api/chat endpoint."""

import httpx

from geoagent.providers.errors import ProviderError, ProviderTimeout, ProviderUnavailable
from geoagent.providers.types import Generation


class OllamaProvider:
    """LLMProvider backed by a (possibly remote) Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        num_ctx: int | None = 8192,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client()

    def close(self) -> None:
        """Close the HTTP client if this provider created it (an injected client is the caller's)."""
        if self._owns_client:
            self._client.close()

    def generate(self, system: str, prompt: str) -> Generation:
        """Run one non-streaming chat completion; failures raise ProviderError subclasses."""
        options: dict[str, float | int] = {"temperature": 0}
        if self.num_ctx is not None:
            # Ollama's default context window (~2-4k tokens) silently drops the
            # start of long RAG prompts, so request an explicit, larger one.
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": options,
        }
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=httpx.Timeout(self.timeout_s, connect=min(self.timeout_s, 5.0)),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Ollama request timed out (timeout={self.timeout_s}s)") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"Ollama unreachable at {self.base_url}: {exc}") from exc

        if response.status_code >= 500:
            raise ProviderUnavailable(
                f"Ollama server error {response.status_code}: {self._error_text(response)}"
            )
        if not response.is_success:
            raise ProviderError(
                f"Ollama request failed ({response.status_code}): {self._error_text(response)}"
            )

        try:
            body = response.json()
            text = body["message"]["content"]
            model = body["model"]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(
                f"unexpected response from Ollama: {self._error_text(response)[:200]}"
            ) from exc
        if body.get("done_reason") == "length":
            raise ProviderError(
                "Ollama response truncated (done_reason=length); raise num_ctx or shorten the prompt"
            )
        return Generation(
            text=text,
            model=model,
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
        )

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            return str(response.json()["error"])
        except (ValueError, KeyError, TypeError):
            return response.text[:300]
