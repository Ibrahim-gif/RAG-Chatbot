# src/rag/llms/openai_llm.py
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

class OpenAIChatLLM:
    """
    Minimal LangChain OpenAI chat wrapper with a single .generate(prompt) method.
    Keeps the pipeline decoupled from LangChain message details.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 1000
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key was provided.")

        self._client = OpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=temperature,
        )
        
    def structured_generate(self, messages: list, response_class) -> any:
        """
        Returns the assistant's text response parsed into the given response_class (a Pydantic model).
        """
        completion = self._client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_class,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message
