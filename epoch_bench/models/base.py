from __future__ import annotations

from abc import ABC, abstractmethod


class ModelProvider(ABC):
    def __init__(self, model: str) -> None:
        self.model = model
        self.temperature: float = 0.0
        self.max_tokens: int = 2048

    @property
    def name(self) -> str:
        return self.model

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...
