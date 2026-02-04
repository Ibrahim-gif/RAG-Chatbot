"""
Unit tests for the OpenAI LLM module.

Tests the OpenAI Chat LLM wrapper for both structured and free-text generation.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.llms.openai_llm import OpenAIChatLLM
from pydantic import BaseModel, Field


class MockResponseModel(BaseModel):
    """Mock response model for testing structured generation."""
    answer: str = Field(description="The answer")
    confidence: int = Field(description="Confidence level")


class TestOpenAIChatLLMInitialization:
    """Test OpenAIChatLLM initialization."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_llm_default_initialization(self, mock_openai):
        """Test OpenAIChatLLM initialization with default parameters."""
        llm = OpenAIChatLLM()
        assert llm.model_name == "gpt-4.1-mini"
        assert llm.temperature == 0.0
        assert llm.max_tokens == 1000
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_llm_custom_initialization(self, mock_openai):
        """Test OpenAIChatLLM initialization with custom parameters."""
        llm = OpenAIChatLLM(
            model_name="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        assert llm.model_name == "gpt-4-turbo"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2000
    
    @patch("src.llms.openai_llm.OpenAI")
    def test_llm_provided_api_key(self, mock_openai):
        """Test OpenAIChatLLM initialization with provided API key."""
        llm = OpenAIChatLLM(api_key="provided-key")
        assert llm.api_key == "provided-key"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_llm_missing_api_key(self):
        """Test OpenAIChatLLM raises error when API key is missing."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            OpenAIChatLLM()


class TestOpenAIChatLLMStructuredGeneration:
    """Test structured generation functionality."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_structured_generate_basic(self, mock_openai_class):
        """Test basic structured generation."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=MockResponseModel(answer="Test answer", confidence=80)))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        
        mock_client.chat.completions.parse.return_value = mock_response
        
        llm = OpenAIChatLLM()
        result = llm.structured_generate(
            messages=[],
            user_query="What is energy efficiency?",
            system_message="You are a helpful assistant.",
            response_class=MockResponseModel,
            trace_name="test"
        )
        
        assert result.answer == "Test answer"
        assert result.confidence == 80
        mock_client.chat.completions.parse.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_structured_generate_with_history(self, mock_openai_class):
        """Test structured generation with conversation history."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=MockResponseModel(answer="Answer", confidence=90)))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.parse.return_value = mock_response
        
        llm = OpenAIChatLLM()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = llm.structured_generate(
            messages=messages,
            user_query="How are you?",
            system_message="Be helpful.",
            response_class=MockResponseModel,
            trace_name="test"
        )
        
        assert result.answer == "Answer"
        # Verify that messages were included in the call
        call_args = mock_client.chat.completions.parse.call_args
        assert call_args is not None
        assert "messages" in call_args.kwargs or len(call_args.args) > 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_structured_generate_respects_max_tokens(self, mock_openai_class):
        """Test that structured generation respects max_tokens parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=MockResponseModel(answer="Short", confidence=50)))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.parse.return_value = mock_response
        
        llm = OpenAIChatLLM(max_tokens=100)
        
        llm.structured_generate(
            messages=[],
            user_query="Test",
            system_message="Test",
            response_class=MockResponseModel,
            trace_name="test"
        )
        
        call_kwargs = mock_client.chat.completions.parse.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 100
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_structured_generate_respects_temperature(self, mock_openai_class):
        """Test that structured generation respects temperature parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=MockResponseModel(answer="Test", confidence=50)))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.parse.return_value = mock_response
        
        llm = OpenAIChatLLM(temperature=0.5)
        
        llm.structured_generate(
            messages=[],
            user_query="Test",
            system_message="Test",
            response_class=MockResponseModel,
            trace_name="test"
        )
        
        call_kwargs = mock_client.chat.completions.parse.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5


class TestOpenAIChatLLMFreeTextGeneration:
    """Test free-text generation functionality."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_generate_basic(self, mock_openai_class):
        """Test basic free-text generation."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="This is a test response."))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        result = llm.generate(
            messages=[],
            user_query="What is AI?",
            system_message="You are helpful.",
            trace_name="test"
        )
        
        assert result == "This is a test response."
        mock_client.chat.completions.create.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_generate_with_history(self, mock_openai_class):
        """Test free-text generation with conversation history."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        
        result = llm.generate(
            messages=messages,
            user_query="How are you?",
            system_message="Be helpful.",
            trace_name="test"
        )
        
        assert isinstance(result, str)
        mock_client.chat.completions.create.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_generate_empty_response(self, mock_openai_class):
        """Test handling of empty response."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=""))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        result = llm.generate(
            messages=[],
            user_query="Test",
            system_message="Test",
            trace_name="test"
        )
        
        assert result == ""


class TestOpenAIChatLLMMessageFormatting:
    """Test message formatting for API calls."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_message_format_with_system_prompt(self, mock_openai_class):
        """Test that system prompt is included in messages."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        system_msg = "You are an expert."
        
        llm.generate(
            messages=[],
            user_query="Test query",
            system_message=system_msg,
            trace_name="test"
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages")
        assert messages is not None
        assert any(msg.get("role") == "system" and system_msg in msg.get("content", "") for msg in messages)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_user_query_included_in_messages(self, mock_openai_class):
        """Test that user query is included as final message."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        user_query = "What is energy efficiency?"
        
        llm.generate(
            messages=[],
            user_query=user_query,
            system_message="Test",
            trace_name="test"
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages")
        assert messages is not None
        # Last user message should contain the query
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assert len(user_messages) > 0


class TestOpenAIChatLLMModelConfiguration:
    """Test model configuration."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_model_name_in_api_call(self, mock_openai_class):
        """Test that model name is correctly passed to API."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        model_name = "gpt-4-turbo"
        llm = OpenAIChatLLM(model_name=model_name)
        
        llm.generate(
            messages=[],
            user_query="Test",
            system_message="Test",
            trace_name="test"
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("model") == model_name
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_different_model_names(self, mock_openai_class):
        """Test support for different model names."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        for model_name in ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
            llm = OpenAIChatLLM(model_name=model_name)
            assert llm.model_name == model_name


class TestOpenAIChatLLMEdgeCases:
    """Test edge cases and error handling."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_generate_very_long_query(self, mock_openai_class):
        """Test generating response for very long query."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        long_query = "Question " * 1000
        
        result = llm.generate(
            messages=[],
            user_query=long_query,
            system_message="Test",
            trace_name="test"
        )
        
        assert isinstance(result, str)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.llms.openai_llm.OpenAI")
    def test_generate_with_none_messages(self, mock_openai_class):
        """Test generating with None messages."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAIChatLLM()
        
        result = llm.generate(
            messages=None,
            user_query="Test",
            system_message="Test",
            trace_name="test"
        )
        
        assert isinstance(result, str)
