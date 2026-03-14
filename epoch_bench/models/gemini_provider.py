from __future__ import annotations

import asyncio

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from epoch_bench.models.base import ModelProvider


class GeminiProvider(ModelProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._model = genai.GenerativeModel(model)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=config,
                )
                return response.text
            except ResourceExhausted:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
