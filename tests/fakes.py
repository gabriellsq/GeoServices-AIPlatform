import hashlib
import math
import random

from geoagent.providers.types import Generation, TaskType

DIM = 768


def unit_vector(seed: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-random unit vector derived from a string."""
    rnd = random.Random(hashlib.sha256(seed.encode()).digest())
    v = [rnd.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def basis_vector(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def normalized(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class FakeEmbedder:
    def __init__(
        self,
        dim: int = DIM,
        fixed: dict[str, list[float]] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.dim = dim
        self.fixed = fixed or {}
        self.fail_with = fail_with
        self.calls: list[tuple[list[str], TaskType]] = []

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        self.calls.append((list(texts), task_type))
        if self.fail_with is not None:
            raise self.fail_with
        return [self.fixed.get(t) or unit_vector(t, self.dim) for t in texts]


class FakeLLM:
    def __init__(
        self, text: str = "answer", model: str = "fake-model", fail_with: Exception | None = None
    ) -> None:
        self.model = model
        self.text = text
        self.fail_with = fail_with
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> Generation:
        self.calls.append((system, prompt))
        if self.fail_with is not None:
            raise self.fail_with
        return Generation(text=self.text, model=self.model, tokens_in=10, tokens_out=5)
