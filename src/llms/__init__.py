"""
Language Model (LLM) module for the RAG system.

Provides wrappers and interfaces for interacting with large language models,
including both structured and free-text generation capabilities.
"""

from .openai_llm import OpenAIChatLLM

__all__ = ["OpenAIChatLLM"]
