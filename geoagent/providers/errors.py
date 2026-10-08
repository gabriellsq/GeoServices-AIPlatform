"""Provider-related exceptions."""


class ProviderError(Exception):
    """Base exception for any provider failure."""

    pass


class ProviderUnavailable(ProviderError):
    """Provider is unavailable (host unreachable, 5xx, rate-limited after retries)."""

    pass


class ProviderTimeout(ProviderError):
    """Provider call exceeded its timeout."""

    pass
