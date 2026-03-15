from __future__ import annotations

import asyncio

import openai

from epoch_bench.models.base import ModelProvider


class OpenAIProvider(ModelProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = openai.AsyncOpenAI()

    _REASONING_PREFIXES = ("o1", "o3", "o4")

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        is_reasoning = any(self.model.startswith(p) for p in self._REASONING_PREFIXES)

        for attempt in range(5):
            try:
                if is_reasoning:
                    # Reasoning models: no temperature/system, use max_completion_tokens
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
                        ],
                        max_completion_tokens=self.max_tokens,
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                return response.choices[0].message.content
            except openai.RateLimitError:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
