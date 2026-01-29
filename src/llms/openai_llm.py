# src/rag/llms/openai_llm.py
from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class OpenAIChatLLM:
    """
    Minimal LangChain OpenAI chat wrapper with a single .generate(prompt) method.
    Keeps your pipeline decoupled from LangChain message details.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 1000,
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key was provided.")

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """
        Returns the assistant's text response.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        resp = self._client.invoke(messages)
        # LangChain returns an AIMessage with `.content`
        return resp.content if hasattr(resp, "content") else str(resp)
