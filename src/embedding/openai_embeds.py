"""
OpenAI embedding wrapper module.

This module provides a wrapper around LangChain's OpenAIEmbeddings for generating
vector embeddings of text using OpenAI's embedding models.

"""

from __future__ import annotations

import os
from typing import List

from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder:
    """
    Wrapper around LangChain OpenAIEmbeddings for generating vector embeddings.
    
    This class provides a simple interface for embedding text documents and queries
    using OpenAI's embedding models.
    
    Attributes:
        model_name (str): The name of the OpenAI embedding model to use.
        api_key (str): The OpenAI API key for authentication.
        _client (OpenAIEmbeddings): Internal LangChain OpenAIEmbeddings client.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str | None = None,
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model_name (str, optional): The embedding model to use. 
                                       Defaults to "text-embedding-3-large".
            api_key (str, optional): OpenAI API key. If not provided, will read from
                                    OPENAI_API_KEY environment variable.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not set and no api_key is provided.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key was provided.")

        self._client = OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
        )
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts (List[str]): A list of text strings to embed.
        
        Returns:
            List[List[float]]: A list of embedding vectors, where each vector is a list of floats.
        
        Example:
            >>> embedder = OpenAIEmbedder()
            >>> embeddings = embedder.embed_documents(["Hello world", "Goodbye world"])
        """
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query.
        
        Args:
            text (str): The query text to embed.
        
        Returns:
            List[float]: The embedding vector as a list of floats.
        
        Example:
            >>> embedder = OpenAIEmbedder()
            >>> embedding = embedder.embed_query("What is energy efficiency?")
        """
        return self._client.embed_query(text)