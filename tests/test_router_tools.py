"""
Tests for router functionality with tools, including fallback scenarios.
"""

import os
import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock, patch

from v_router import Client, LLM, BackupModel
from v_router.classes.tools import ToolCall, Tools
from v_router.classes.message import Message
from v_router.classes.response import Response, Content, Usage
from v_router.router import Router


class MockToolSchema(BaseModel):
    """Mock tool schema for testing."""
    param1: str = Field(..., description="First parameter")
    param2: int = Field(42, description="Second parameter")


class TestRouterToolInheritance:
    """Test that router properly handles tool inheritance in fallback scenarios."""
    
    def _create_mock_tools(self):
        """Create mock tools for testing."""
        return Tools(tools=[
            ToolCall(
                name="mock_tool",
                description="A mock tool for testing",
                input_schema=MockToolSchema.model_json_schema()
            )
        ])
    
    def test_router_initialization_with_tools(self):
        """Test router initialization with tools."""
        tools = self._create_mock_tools()
        llm_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=tools
        )
        
        router = Router(llm_config)
        assert router.primary_config.tools is not None
        assert len(router.primary_config.tools.tools) == 1
    
    def test_backup_model_tool_inheritance_logic(self):
        """Test the logic for tool inheritance in backup models."""
        primary_tools = self._create_mock_tools()
        
        # Backup model without tools (should inherit)
        backup_without_tools = LLM(
            model_name="claude-sonnet-3.5",
            provider="anthropic"
            # No tools specified
        )
        
        # Backup model with its own tools (should not inherit)
        backup_with_tools = LLM(
            model_name="gemini-pro",
            provider="google",
            tools=Tools(tools=[
                ToolCall(
                    name="backup_specific_tool",
                    description="Tool specific to backup",
                    input_schema={"type": "object"}
                )
            ])
        )
        
        primary_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=primary_tools,
            backup_models=[
                BackupModel(model=backup_without_tools, priority=1),
                BackupModel(model=backup_with_tools, priority=2)
            ]
        )
        
        # Test the inheritance logic
        ordered_backups = primary_config.get_ordered_backup_models()
        
        # First backup has no tools (should inherit)
        assert ordered_backups[0].tools is None
        
        # Second backup has its own tools (should not inherit)
        assert ordered_backups[1].tools is not None
        assert ordered_backups[1].tools.tools[0].name == "backup_specific_tool"
    
    @pytest.mark.asyncio
    async def test_router_tool_inheritance_with_mock_provider(self):
        """Test router tool inheritance using mocked providers."""
        tools = self._create_mock_tools()
        
        # Create a primary config with tools
        primary_config = LLM(
            model_name="failing-model",
            provider="anthropic",
            tools=tools,
            backup_models=[
                BackupModel(
                    model=LLM(
                        model_name="gpt-4.1-nano",
                        provider="openai"
                        # No tools - should inherit from primary
                    ),
                    priority=1
                )
            ]
        )
        
        router = Router(primary_config)
        
        # Mock the provider instances
        mock_anthropic_provider = AsyncMock()
        mock_openai_provider = AsyncMock()
        
        # Make primary provider fail
        mock_anthropic_provider.create_message.side_effect = Exception("Primary model failed")
        
        # Make backup provider succeed and capture the tools it receives
        captured_tools = None
        async def capture_tools(*args, **kwargs):
            nonlocal captured_tools
            captured_tools = kwargs.get('tools')
            return Response(
                content=[Content(type="text", role="assistant", text="Success with tools")],
                model="gpt-4",
                provider="openai",
                usage=Usage(input_tokens=10, output_tokens=5),
                raw_response={}
            )
        
        mock_openai_provider.create_message = capture_tools
        
        # Patch the provider instances
        router._provider_instances = {
            "anthropic": mock_anthropic_provider,
            "openai": mock_openai_provider
        }
        
        # Create test messages
        messages = [Message(role="user", content="Test message")]
        
        # Route the request - should fail on primary and succeed on backup
        response = await router.route_request(messages)
        
        # Verify the backup succeeded
        assert len(response.content) == 1
        assert response.content[0].text == "Success with tools"
        assert response.provider == "openai"
        
        # Verify that tools were inherited by the backup model
        assert captured_tools is not None
        assert len(captured_tools.tools) == 1
        assert captured_tools.tools[0].name == "mock_tool"


class TestRouterToolFormatConversion:
    """Test that router properly converts tools to provider-specific formats."""
    
    def _create_test_tools(self):
        """Create test tools."""
        return Tools(tools=[
            ToolCall(
                name="test_function",
                description="A test function",
                input_schema={
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "First argument"},
                        "arg2": {"type": "number", "description": "Second argument"}
                    },
                    "required": ["arg1"]
                }
            )
        ])
    
    @pytest.mark.asyncio
    async def test_anthropic_tool_format_conversion(self):
        """Test tool format conversion for Anthropic provider."""
        tools = self._create_test_tools()
        
        with patch('v_router.providers.anthropic.AsyncAnthropic') as mock_anthropic:
            # Mock the client and response
            mock_client = AsyncMock()
            mock_anthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Test response"
            mock_response.model = "claude-sonnet-4"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            
            mock_client.messages.create.return_value = mock_response
            
            # Create provider and test
            from v_router.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider()
            
            messages = [Message(role="user", content="Test")]
            await provider.create_message(messages, "claude-sonnet-4", tools=tools)
            
            # Verify the call was made with properly formatted tools
            call_args = mock_client.messages.create.call_args
            assert 'tools' in call_args.kwargs
            
            anthropic_tools = call_args.kwargs['tools']
            assert len(anthropic_tools) == 1
            assert anthropic_tools[0]['name'] == 'test_function'
            assert anthropic_tools[0]['description'] == 'A test function'
            assert 'input_schema' in anthropic_tools[0]
    
    @pytest.mark.asyncio
    async def test_openai_tool_format_conversion(self):
        """Test tool format conversion for OpenAI provider."""
        tools = self._create_test_tools()
        
        with patch('v_router.providers.openai.AsyncOpenAI') as mock_openai:
            # Mock the client and response
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.model = "gpt-4"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create provider and test
            from v_router.providers.openai import OpenAIProvider
            provider = OpenAIProvider()
            
            messages = [Message(role="user", content="Test")]
            await provider.create_message(messages, "gpt-4", tools=tools)
            
            # Verify the call was made with properly formatted tools
            call_args = mock_client.chat.completions.create.call_args
            assert 'tools' in call_args.kwargs
            
            openai_tools = call_args.kwargs['tools']
            assert len(openai_tools) == 1
            assert openai_tools[0]['type'] == 'function'
            assert openai_tools[0]['function']['name'] == 'test_function'
            assert openai_tools[0]['function']['description'] == 'A test function'
            assert 'parameters' in openai_tools[0]['function']
    
    @pytest.mark.asyncio
    async def test_google_tool_format_conversion(self):
        """Test tool format conversion for Google provider."""
        tools = self._create_test_tools()
        
        with patch('v_router.providers.google.genai') as mock_genai:
            # Mock the client and response
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = "Test response"
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 5
            mock_response.usage_metadata.total_token_count = 15
            
            mock_client.models.generate_content.return_value = mock_response
            
            # Mock the types
            mock_genai.types.GenerateContentConfig = MagicMock()
            mock_genai.types.Tool = MagicMock()
            
            # Create provider and test
            from v_router.providers.google import GoogleProvider
            provider = GoogleProvider()
            
            messages = [Message(role="user", content="Test")]
            await provider.create_message(messages, "gemini-pro", tools=tools)
            
            # Verify Tool was called with function declarations
            mock_genai.types.Tool.assert_called_once()
            call_args = mock_genai.types.Tool.call_args
            assert 'function_declarations' in call_args.kwargs
            
            function_declarations = call_args.kwargs['function_declarations']
            assert len(function_declarations) == 1
            assert function_declarations[0]['name'] == 'test_function'


class TestClientToolIntegration:
    """Test client-level tool integration."""
    
    @pytest.mark.asyncio
    async def test_client_tool_passing(self):
        """Test that client properly passes tools to router."""
        tools = Tools(tools=[
            ToolCall(
                name="client_test_tool",
                description="Tool for client testing",
                input_schema={"type": "object"}
            )
        ])
        
        llm_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=tools,
            max_tokens=100
        )
        
        client = Client(llm_config)
        
        # Verify the client has the tools configured
        assert client.router.primary_config.tools is not None
        assert len(client.router.primary_config.tools.tools) == 1
        assert client.router.primary_config.tools.tools[0].name == "client_test_tool"
    
    @pytest.mark.asyncio
    async def test_client_with_backup_model_tools(self):
        """Test client with backup models that have different tool configurations."""
        primary_tools = Tools(tools=[
            ToolCall(name="primary_tool", description="Primary tool", input_schema={"type": "object"})
        ])
        
        backup_tools = Tools(tools=[
            ToolCall(name="backup_tool", description="Backup tool", input_schema={"type": "object"})
        ])
        
        llm_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=primary_tools,
            backup_models=[
                BackupModel(
                    model=LLM(
                        model_name="claude-3-5-haiku",
                        provider="anthropic",
                        tools=backup_tools
                    ),
                    priority=1
                ),
                BackupModel(
                    model=LLM(
                        model_name="gemini-2.5-flash-preview",
                        provider="google"
                        # No tools - should inherit from primary
                    ),
                    priority=2
                )
            ]
        )
        
        client = Client(llm_config)
        
        # Verify primary tools
        assert client.router.primary_config.tools.tools[0].name == "primary_tool"
        
        # Verify backup model tool configurations
        ordered_backups = client.router.primary_config.get_ordered_backup_models()
        assert ordered_backups[0].tools.tools[0].name == "backup_tool"  # Has its own tools
        assert ordered_backups[1].tools is None  # Should inherit from primary


# Integration tests that require API keys
class TestRealAPIToolCalling:
    """Integration tests with real API calls."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_tool_inheritance_fallback(self):
        """Test tool inheritance in a real fallback scenario."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        tools = Tools(tools=[
            ToolCall(
                name="get_info",
                description="Get information about something",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to get info about"}
                    },
                    "required": ["query"]
                }
            )
        ])
        
        # Use a model that will definitely fail
        llm_config = LLM(
            model_name="definitely-nonexistent-model",
            provider="anthropic",
            tools=tools,
            backup_models=[
                BackupModel(
                    model=LLM(
                        model_name="gpt-4.1-nano",
                        provider="openai",
                        max_tokens=100
                        # No tools specified - should inherit
                    ),
                    priority=1
                )
            ]
        )
        
        client = Client(llm_config)
        
        try:
            response = await client.messages.create(
                messages=[
                    {"role": "user", "content": "Get info about Python programming"}
                ]
            )
            
            # Should have fallen back to OpenAI
            assert response.provider == "openai"
            
            # Check if tools were available (OpenAI might or might not use them)
            # The important thing is that no error occurred due to missing tools
            assert response.content is not None
            
        except Exception as e:
            # If it fails, it should not be due to missing tools
            assert "tools" not in str(e).lower()
            # Re-raise if it's a different error
            raise
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_multiple_provider_tool_consistency(self):
        """Test that the same tools work consistently across different providers."""
        available_providers = []
        if os.getenv("ANTHROPIC_API_KEY"):
            available_providers.append(("anthropic", "claude-3-5-haiku"))
        if os.getenv("OPENAI_API_KEY"):
            available_providers.append(("openai", "gpt-4.1-nano"))
        if os.getenv("GEMINI_API_KEY"):
            available_providers.append(("google", "gemini-2.5-flash-preview"))
        
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers configured to test consistency")
        
        tools = Tools(tools=[
            ToolCall(
                name="simple_calculator",
                description="Perform simple math",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            )
        ])
        
        responses = {}
        
        for provider_name, model_name in available_providers[:2]:  # Test first 2 available
            llm_config = LLM(
                model_name=model_name,
                provider=provider_name,
                tools=tools,
                max_tokens=150
            )
            
            client = Client(llm_config)
            response = await client.messages.create(
                messages=[
                    {"role": "user", "content": "Calculate 7 times 8"}
                ]
            )
            
            responses[provider_name] = response
        
        # Both providers should have received the request successfully
        for provider_name, response in responses.items():
            assert response.provider == provider_name
            assert response.content is not None
            
            # Check for tool usage - all providers now use the same format
            if len(response.tool_use) > 0:
                assert response.tool_use[0].name == "simple_calculator"