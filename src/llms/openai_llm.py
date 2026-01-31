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

        self._client = OpenAI()
        
    def structured_generate(self, messages: list, user_query: str, system_message:str, response_class) -> any:
        """
        Returns the assistant's text response parsed into the given response_class (a Pydantic model).
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        print(f"OpenAIChatLLM structured_generate messages: {messages}")
        response = self._client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_class,
            max_tokens=self.max_tokens,
        )
        print(f"OpenAIChatLLM structured_generate response: {response}")
        return response.choices[0].message.parsed

    def generate(self, messages: list, user_query: str, system_message:str) -> str:
        """
        Returns the assistant's text response as a string.
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        print(f"OpenAIChatLLM generate response: {response}")
        return response.choices[0].message.content