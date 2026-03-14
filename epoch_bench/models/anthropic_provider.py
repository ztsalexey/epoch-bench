from __future__ import annotations

import asyncio

import anthropic

from epoch_bench.models.base import ModelProvider


class AnthropicProvider(ModelProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = anthropic.AsyncAnthropic()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(5):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
