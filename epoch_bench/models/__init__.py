from __future__ import annotations

from epoch_bench.models.anthropic_provider import AnthropicProvider
from epoch_bench.models.base import ModelProvider
from epoch_bench.models.openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "ModelProvider",
    "OpenAIProvider",
    "get_provider",
]


def get_provider(provider: str, model: str) -> ModelProvider:
    # Lazy imports for providers with heavy/deprecated dependencies
    if provider == "gemini":
        from epoch_bench.models.gemini_provider import GeminiProvider
        return GeminiProvider(model)
    if provider == "deepseek":
        from epoch_bench.models.deepseek_provider import DeepSeekProvider
        return DeepSeekProvider(model)

    providers: dict[str, type[ModelProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown provider {provider!r}, expected one of "
            f"{['openai', 'anthropic', 'gemini', 'deepseek']}"
        )
    return providers[provider](model)
