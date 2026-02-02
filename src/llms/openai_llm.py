# src/rag/llms/openai_llm.py
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
    
    @traceable(name="openai.chat.completions.parse", run_type="llm", metadata={"ls_provider": "openai"})
    def structured_generate(self, messages: list, user_query: str, system_message:str, response_class, trace_name: str) -> any:
        """
        Returns the assistant's text response parsed into the given response_class (a Pydantic model).
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        t0 = time.perf_counter()
        response = self._client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=response_class,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        latency_s = time.perf_counter() - t0
        logging.info(f"OpenAIChatLLM structured_generate response: {response}")
        
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        
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
        Returns the assistant's text response as a string.
        """
        messages = [{"role": "system", "content": system_message}] + messages + [{"role": "user", "content": f"User Query: {user_query}"}]
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        latency_s = time.perf_counter() - t0

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        
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
        Attach token usage in LangSmith-recognized format: usage_metadata. :contentReference[oaicite:3]{index=3}
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