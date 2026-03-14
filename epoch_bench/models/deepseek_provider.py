from __future__ import annotations

import os

import openai

from epoch_bench.models.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
