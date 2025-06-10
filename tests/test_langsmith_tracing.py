"""Tests for LangSmith tracing integration."""

import os
from unittest.mock import Mock, patch, MagicMock
import sys

import pytest

from v_router.providers.openai import OpenAIProvider, AzureOpenAIProvider
from v_router.providers.anthropic import AnthropicProvider


class TestLangSmithTracing:
    """Test LangSmith tracing integration."""

    def test_openai_provider_without_tracing(self):
        """Test OpenAI provider without LangSmith tracing."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(api_key="test-key")
            # Client should not be wrapped
            assert provider.client.__class__.__name__ == "AsyncOpenAI"

    def test_openai_provider_with_tracing_disabled(self):
        """Test OpenAI provider with LangSmith tracing explicitly disabled."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}):
            provider = OpenAIProvider(api_key="test-key")
            # Client should not be wrapped
            assert provider.client.__class__.__name__ == "AsyncOpenAI"

    def test_openai_provider_with_tracing_enabled_no_langsmith(self):
        """Test OpenAI provider with tracing enabled but LangSmith not installed."""
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            # Mock the import failure
            with patch.dict('sys.modules', {'langsmith': None, 'langsmith.wrappers': None}):
                provider = OpenAIProvider(api_key="test-key")
                # Should fall back to unwrapped client
                assert provider.client.__class__.__name__ == "AsyncOpenAI"

    def test_openai_provider_with_tracing_enabled_with_langsmith(self):
        """Test OpenAI provider with tracing enabled and LangSmith available."""
        mock_wrapped_client = Mock()
        mock_wrap_openai = Mock(return_value=mock_wrapped_client)
        
        # Mock the langsmith module
        mock_langsmith = MagicMock()
        mock_langsmith.wrappers.wrap_openai = mock_wrap_openai
        
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            with patch.dict('sys.modules', {'langsmith': mock_langsmith, 'langsmith.wrappers': mock_langsmith.wrappers}):
                provider = OpenAIProvider(api_key="test-key")
                
                # Should call wrap_openai
                mock_wrap_openai.assert_called_once()
                # Should use wrapped client
                assert provider.client == mock_wrapped_client

    def test_azure_openai_provider_without_tracing(self):
        """Test Azure OpenAI provider without LangSmith tracing."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AzureOpenAIProvider(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/"
            )
            # Client should not be wrapped
            assert provider.client.__class__.__name__ == "AsyncAzureOpenAI"

    def test_azure_openai_provider_with_tracing_enabled_with_langsmith(self):
        """Test Azure OpenAI provider with tracing enabled and LangSmith available."""
        mock_wrapped_client = Mock()
        mock_wrap_openai = Mock(return_value=mock_wrapped_client)
        
        # Mock the langsmith module
        mock_langsmith = MagicMock()
        mock_langsmith.wrappers.wrap_openai = mock_wrap_openai
        
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            with patch.dict('sys.modules', {'langsmith': mock_langsmith, 'langsmith.wrappers': mock_langsmith.wrappers}):
                provider = AzureOpenAIProvider(
                    api_key="test-key",
                    azure_endpoint="https://test.openai.azure.com/"
                )
                
                # Should call wrap_openai
                mock_wrap_openai.assert_called_once()
                # Should use wrapped client
                assert provider.client == mock_wrapped_client

    def test_anthropic_provider_without_tracing(self):
        """Test Anthropic provider without LangSmith tracing."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AnthropicProvider(api_key="test-key")
            # Client should not be wrapped
            assert provider.client.__class__.__name__ == "AsyncAnthropic"

    def test_anthropic_provider_with_tracing_enabled_with_langsmith(self):
        """Test Anthropic provider with tracing enabled and LangSmith available."""
        mock_wrapped_client = Mock()
        mock_wrap_anthropic = Mock(return_value=mock_wrapped_client)
        
        # Mock the langsmith module
        mock_langsmith = MagicMock()
        mock_langsmith.wrappers.wrap_anthropic = mock_wrap_anthropic
        
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            with patch.dict('sys.modules', {'langsmith': mock_langsmith, 'langsmith.wrappers': mock_langsmith.wrappers}):
                provider = AnthropicProvider(api_key="test-key")
                
                # Should call wrap_anthropic
                mock_wrap_anthropic.assert_called_once()
                # Should use wrapped client
                assert provider.client == mock_wrapped_client

    def test_tracing_case_insensitive(self):
        """Test that tracing environment variable is case sensitive (only 'true' enables)."""
        test_cases = ["True", "TRUE", "yes", "1", "on"]
        
        for value in test_cases:
            with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": value}):
                provider = OpenAIProvider(api_key="test-key")
                # Only exact "true" should enable tracing
                assert provider.client.__class__.__name__ == "AsyncOpenAI"

    def test_tracing_exact_true_value(self):
        """Test that only exact 'true' value enables tracing."""
        mock_wrapped_client = Mock()
        mock_wrap_openai = Mock(return_value=mock_wrapped_client)
        
        # Mock the langsmith module
        mock_langsmith = MagicMock()
        mock_langsmith.wrappers.wrap_openai = mock_wrap_openai
        
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            with patch.dict('sys.modules', {'langsmith': mock_langsmith, 'langsmith.wrappers': mock_langsmith.wrappers}):
                provider = OpenAIProvider(api_key="test-key")
                assert provider.client == mock_wrapped_client