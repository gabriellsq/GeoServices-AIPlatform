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
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = client if client is not None else httpx.Client()

    def generate(self, system: str, prompt: str) -> Generation:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Ollama request timed out after {self.timeout_s}s") from exc
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

        body = response.json()
        return Generation(
            text=body["message"]["content"],
            model=body["model"],
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
        )

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            return str(response.json()["error"])
        except (ValueError, KeyError, TypeError):
            return response.text
