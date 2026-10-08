"""Provider protocol definitions and core types."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class TaskType(StrEnum):
    """Task type classification for embedding requests."""

    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


@dataclass(frozen=True)
class Generation:
    """LLM generation output with metadata."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    model: str

    def generate(self, system: str, prompt: str) -> Generation:
        """Generate a response given system and prompt."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    dim: int

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        """Embed texts, returning L2-normalised vectors in input order."""
        ...
