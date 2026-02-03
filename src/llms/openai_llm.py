# src/rag/llms/openai_llm.py
"""
OpenAI Chat LLM wrapper module.

This module provides a wrapper around OpenAI's Chat Completion API with support for
both free-text generation and structured generation with Pydantic model validation.
Includes LangSmith integration for tracing and observability.

"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree 
import time
import logging

class OpenAIChatLLM:
    """
    OpenAI Chat LLM wrapper with structured and free-text generation.
    
    Provides a clean interface to OpenAI's Chat Completions API with support for:
    - Free-text generation with optional system/user message templates
    - Structured generation with Pydantic model validation
    - LangSmith tracing and observability
    - Token usage tracking
    - Deterministic defaults (temperature=0)
    
    Attributes:
        model_name (str): Name of the OpenAI model to use (e.g., 'gpt-4-turbo').
        api_key (str): OpenAI API key for authentication.
        max_tokens (int): Maximum tokens in the generated response.
        temperature (float): Controls response randomness (0.0 = deterministic).
        _client (OpenAI): OpenAI client instance.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1000
    ):
        """
        Initialize the OpenAI Chat LLM.
        
        Args:
            model_name (str, optional): OpenAI model name. Defaults to "gpt-4.1-mini".
            api_key (str, optional): OpenAI API key. If not provided, reads from
                                    OPENAI_API_KEY environment variable.
            temperature (float, optional): Response randomness (0.0-2.0). Defaults to 0.0.
            max_tokens (int, optional): Maximum tokens in response. Defaults to 1000.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not set and no api_key is provided.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key was provided.")

        self._client = OpenAI()
    
    @traceable(name="openai.chat.completions.parse", run_type="llm", metadata={"ls_provider": "openai"})
    def structured_generate(self, messages: list, user_query: str, system_message:str, response_class, trace_name: str) -> any:
        """
        Generate a structured response parsed into a Pydantic model.
        
        Uses OpenAI's structured output API to ensure responses conform to the
        provided Pydantic response class schema.
        
        Args:
            messages (list): Conversation history as a list of message dicts with 'role' and 'content'.
            user_query (str): The current user query or request.
            system_message (str): The system prompt/instructions for the LLM.
            response_class: A Pydantic model class to parse the response into.
            trace_name (str): Name for this operation in LangSmith tracing.
        
        Returns:
            The parsed response as an instance of response_class.
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        t0 = time.perf_counter()
        response = self._client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_class,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_s = time.perf_counter() - t0
        logging.info(f"OpenAIChatLLM structured_generate response: {response}")
        
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        
        # attach token usage to LangSmith run
        self._attach_langsmith_usage(prompt_tokens, completion_tokens)

        # attach latency to the LangSmith run metadata (easy to filter/chart)
        run = get_current_run_tree()
        if run is not None and trace_name:
            run.name = trace_name
            run.metadata["latency"] = latency_s
            run.metadata["ls_model_name"] = self.model_name
        
        return response.choices[0].message.parsed

    def generate(self, messages: list, user_query: str, system_message:str, trace_name: str) -> str:
        """
        Generate a free-text response from the LLM.
        
        Args:
            messages (list): Conversation history as a list of message dicts with 'role' and 'content'.
            user_query (str): The current user query or request.
            system_message (str): The system prompt/instructions for the LLM.
            trace_name (str): Name for this operation in LangSmith tracing.
        
        Returns:
            str: The generated response text from the LLM.
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_s = time.perf_counter() - t0

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        
        self._attach_langsmith_usage(prompt_tokens, completion_tokens)

        run = get_current_run_tree()
        if run is not None and trace_name:
            run.name = trace_name
            run.metadata["latency_s"] = latency_s
            run.metadata["ls_model_name"] = self.model_name
        
        logging.info(f"OpenAIChatLLM generate response: {response}")
        return response.choices[0].message.content
        
    def _attach_langsmith_usage(self, prompt_tokens: Optional[int], completion_tokens: Optional[int]):
        """
        Attach token usage metrics to the LangSmith run trace.
        
        Records input tokens, output tokens, and total tokens in LangSmith-recognized
        format for observability and cost tracking.
        
        Args:
            prompt_tokens (Optional[int]): Number of tokens in the prompt/input.
            completion_tokens (Optional[int]): Number of tokens in the completion/output.
        """
        if prompt_tokens is None and completion_tokens is None:
            return

        usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
        if prompt_tokens is not None and completion_tokens is not None:
            usage_metadata["total_tokens"] = prompt_tokens + completion_tokens

        run = get_current_run_tree()
        if run is not None:
            run.set(usage_metadata=usage_metadata)