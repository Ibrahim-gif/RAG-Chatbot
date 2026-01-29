from __future__ import annotations

import os
from typing import List

from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder:
    """Small wrapper around LangChain OpenAIEmbeddings."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key was provided.")

        self._client = OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
        )
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)