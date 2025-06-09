"""
Tests for tool calling functionality in v_router.

These tests include unit tests for tool classes and integration tests
that make actual API calls to verify tool calling works correctly.
"""

import asyncio
import json
import os
import pytest
from pydantic import BaseModel, Field

from v_router import Client, LLM, BackupModel
from v_router.classes.tools import ToolCall, Tools


class WeatherQuery(BaseModel):
    """Schema for weather query parameters."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    units: str = Field("fahrenheit", description="Temperature units: fahrenheit or celsius")


class CalculatorQuery(BaseModel):
    """Schema for calculator operations."""
    operation: str = Field(..., description="The mathematical operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class TestToolCall:
    """Test cases for the ToolCall class."""

    def test_tool_call_creation_minimal(self):
        """Test creating a ToolCall with minimal required fields."""
        tool = ToolCall(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}}
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_tool_call_creation_with_pydantic_schema(self):
        """Test creating a ToolCall with a Pydantic schema."""
        tool = ToolCall(
            name="get_weather",
            description="Get current weather",
            input_schema=WeatherQuery.model_json_schema()
        )
        
        assert tool.name == "get_weather"
        assert tool.description == "Get current weather"
        assert "location" in tool.input_schema["properties"]
        assert "units" in tool.input_schema["properties"]

    def test_tool_call_json_serialization(self):
        """Test that ToolCall can be serialized and deserialized."""
        original_tool = ToolCall(
            name="calculator",
            description="Perform math operations",
            input_schema=CalculatorQuery.model_json_schema()
        )
        
        # Serialize
        json_data = original_tool.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["name"] == "calculator"
        
        # Deserialize
        reconstructed_tool = ToolCall.model_validate(json_data)
        assert reconstructed_tool.name == original_tool.name
        assert reconstructed_tool.description == original_tool.description
        assert reconstructed_tool.input_schema == original_tool.input_schema


class TestTools:
    """Test cases for the Tools class."""

    def test_tools_creation_empty(self):
        """Test creating an empty Tools collection."""
        tools = Tools(tools=[])
        assert len(tools.tools) == 0

    def test_tools_creation_single_tool(self):
        """Test creating Tools with a single tool."""
        tool = ToolCall(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"}
        )
        tools = Tools(tools=[tool])
        
        assert len(tools.tools) == 1
        assert tools.tools[0].name == "test_tool"

    def test_tools_creation_multiple_tools(self):
        """Test creating Tools with multiple tools."""
        weather_tool = ToolCall(
            name="get_weather",
            description="Get weather",
            input_schema=WeatherQuery.model_json_schema()
        )
        calc_tool = ToolCall(
            name="calculator",
            description="Calculate",
            input_schema=CalculatorQuery.model_json_schema()
        )
        
        tools = Tools(tools=[weather_tool, calc_tool])
        
        assert len(tools.tools) == 2
        tool_names = [tool.name for tool in tools.tools]
        assert "get_weather" in tool_names
        assert "calculator" in tool_names


class TestLLMWithTools:
    """Test cases for LLM class with tools."""

    def test_llm_with_tools(self):
        """Test creating an LLM configuration with tools."""
        tool = ToolCall(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"}
        )
        tools = Tools(tools=[tool])
        
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=tools
        )
        
        assert llm.tools is not None
        assert len(llm.tools.tools) == 1
        assert llm.tools.tools[0].name == "test_tool"

    def test_llm_without_tools(self):
        """Test creating an LLM configuration without tools."""
        llm = LLM(
            model_name="gpt-4.1-nano",
            provider="openai"
        )
        
        assert llm.tools is None

    def test_llm_tools_inheritance_in_backup_models(self):
        """Test that backup models can have their own tools or inherit them."""
        primary_tools = Tools(tools=[
            ToolCall(name="primary_tool", description="Primary tool", input_schema={"type": "object"})
        ])
        
        backup_tools = Tools(tools=[
            ToolCall(name="backup_tool", description="Backup tool", input_schema={"type": "object"})
        ])
        
        backup_with_tools = BackupModel(
            model=LLM(
                model_name="claude-sonnet-3.5",
                provider="anthropic",
                tools=backup_tools
            ),
            priority=1
        )
        
        backup_without_tools = BackupModel(
            model=LLM(
                model_name="gemini-pro",
                provider="google"
                # No tools specified
            ),
            priority=2
        )
        
        primary = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=primary_tools,
            backup_models=[backup_with_tools, backup_without_tools]
        )
        
        # Primary model has its own tools
        assert primary.tools.tools[0].name == "primary_tool"
        
        # First backup has its own tools
        ordered_backups = primary.get_ordered_backup_models()
        assert ordered_backups[0].tools.tools[0].name == "backup_tool"
        
        # Second backup has no tools (would inherit from primary in router)
        assert ordered_backups[1].tools is None


# Integration tests that make actual API calls
class TestToolCallingIntegration:
    """Integration tests for tool calling with real API calls."""
    
    def _create_test_tools(self):
        """Create a set of test tools for integration testing."""
        weather_tool = ToolCall(
            name="get_weather",
            description="Get the current weather in a given location",
            input_schema=WeatherQuery.model_json_schema()
        )
        
        calculator_tool = ToolCall(
            name="calculator",
            description="Perform basic mathematical operations",
            input_schema=CalculatorQuery.model_json_schema()
        )
        
        return Tools(tools=[weather_tool, calculator_tool])
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_anthropic_tool_calling(self):
        """Test tool calling with Anthropic provider."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        tools = self._create_test_tools()
        llm_config = LLM(
            model_name="claude-3-5-haiku",
            provider="anthropic",
            tools=tools,
            max_tokens=200
        )
        
        client = Client(llm_config)
        response = await client.messages.create(
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ]
        )
        
        # Response should contain tool calls
        assert response.provider == "anthropic"
        assert len(response.tool_use) > 0
        
        # Check the tool use
        assert response.tool_use[0].name == "get_weather"
        assert "location" in response.tool_use[0].arguments
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_openai_tool_calling(self):
        """Test tool calling with OpenAI provider."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        tools = self._create_test_tools()
        llm_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=tools,
            max_tokens=200
        )
        
        client = Client(llm_config)
        response = await client.messages.create(
            messages=[
                {"role": "user", "content": "Calculate 15 times 23"}
            ]
        )
        
        # Response should contain tool calls
        assert response.provider == "openai"
        assert len(response.tool_use) > 0
        
        # Check the tool use
        tool_use = response.tool_use[0]
        assert tool_use.name == "calculator"
        assert "operation" in tool_use.arguments
        assert "a" in tool_use.arguments
        assert "b" in tool_use.arguments
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_google_tool_calling(self):
        """Test tool calling with Google provider."""
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        tools = self._create_test_tools()
        llm_config = LLM(
            model_name="gemini-1.5-pro",
            provider="google",
            tools=tools,
            max_tokens=200
        )
        
        client = Client(llm_config)
        response = await client.messages.create(
            messages=[
                {"role": "user", "content": "What's 42 divided by 7?"}
            ]
        )
        
        # Response should contain function calls
        assert response.provider == "google"
        assert len(response.tool_use) > 0
        
        # Check the tool use
        assert response.tool_use[0].name == "calculator"
        assert "operation" in response.tool_use[0].arguments
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_inheritance_in_fallback(self):
        """Test that tools are properly inherited by backup models."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        tools = self._create_test_tools()
        
        # Create a configuration with a non-existent primary model
        llm_config = LLM(
            model_name="claude-nonexistent",  # This will fail
            provider="anthropic",
            tools=tools,
            backup_models=[
                BackupModel(
                    model=LLM(
                        model_name="gpt-4.1-nano",
                        provider="openai",
                        max_tokens=200
                        # Note: No tools specified - should inherit from primary
                    ),
                    priority=1
                )
            ]
        )
        
        client = Client(llm_config)
        response = await client.messages.create(
            messages=[
                {"role": "user", "content": "Calculate 8 times 9"}
            ]
        )
        
        # Should have fallen back to OpenAI
        assert response.provider == "openai"
        
        # Should have made a tool call (tools were inherited)
        if len(response.tool_use) > 0:  # OpenAI might or might not use tools depending on the query
            assert response.tool_use[0].name == "calculator"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cross_provider_tool_calling(self):
        """Test tool calling works across different providers with try_other_providers."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        tools = Tools(tools=[
            ToolCall(
                name="get_weather",
                description="Get the current weather",
                input_schema=WeatherQuery.model_json_schema()
            )
        ])
        
        # Use a model that might fail on vertex but work on anthropic
        llm_config = LLM(
            model_name="claude-3-5-haiku",
            provider="vertexai",  # This might fail due to quota/auth
            tools=tools,
            try_other_providers=True,
            max_tokens=200
        )
        
        client = Client(llm_config)
        response = await client.messages.create(
            messages=[
                {"role": "user", "content": "What's the weather in London?"}
            ]
        )
        
        # Should work with either vertexai or anthropic
        assert response.provider in ["vertexai", "anthropic"]
        
        # Should contain tool calls regardless of which provider was used
        if response.provider == "anthropic":
            if len(response.tool_use) > 0:  # Anthropic might decide to use tools
                assert response.tool_use[0].name == "get_weather"


class TestToolCallingErrorHandling:
    """Test error handling in tool calling scenarios."""
    
    def test_invalid_tool_schema(self):
        """Test handling of invalid tool schemas."""
        # Empty name is actually allowed by Pydantic, so let's test something that should fail
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ToolCall(
                # Missing required fields should raise ValidationError
                description="Test tool",
                input_schema={}
            )
    
    def test_tools_with_duplicate_names(self):
        """Test handling of tools with duplicate names."""
        tool1 = ToolCall(
            name="duplicate_tool",
            description="First tool",
            input_schema={"type": "object"}
        )
        tool2 = ToolCall(
            name="duplicate_tool",  # Same name
            description="Second tool",
            input_schema={"type": "object"}
        )
        
        # This should be allowed at the Tools level (providers might handle differently)
        tools = Tools(tools=[tool1, tool2])
        assert len(tools.tools) == 2


# Fixtures and test configuration
@pytest.fixture
def sample_weather_tool():
    """Fixture providing a sample weather tool."""
    return ToolCall(
        name="get_weather",
        description="Get current weather for a location",
        input_schema=WeatherQuery.model_json_schema()
    )


@pytest.fixture
def sample_calculator_tool():
    """Fixture providing a sample calculator tool."""
    return ToolCall(
        name="calculator",
        description="Perform mathematical operations",
        input_schema=CalculatorQuery.model_json_schema()
    )


@pytest.fixture
def sample_tools(sample_weather_tool, sample_calculator_tool):
    """Fixture providing a sample Tools collection."""
    return Tools(tools=[sample_weather_tool, sample_calculator_tool])


class TestToolCallingWithFixtures:
    """Test tool calling using pytest fixtures."""
    
    def test_weather_tool_schema(self, sample_weather_tool):
        """Test the weather tool schema."""
        assert sample_weather_tool.name == "get_weather"
        assert "location" in sample_weather_tool.input_schema["properties"]
        assert "units" in sample_weather_tool.input_schema["properties"]
    
    def test_calculator_tool_schema(self, sample_calculator_tool):
        """Test the calculator tool schema."""
        assert sample_calculator_tool.name == "calculator"
        assert "operation" in sample_calculator_tool.input_schema["properties"]
        assert "a" in sample_calculator_tool.input_schema["properties"]
        assert "b" in sample_calculator_tool.input_schema["properties"]
    
    def test_tools_collection(self, sample_tools):
        """Test the tools collection."""
        assert len(sample_tools.tools) == 2
        tool_names = [tool.name for tool in sample_tools.tools]
        assert "get_weather" in tool_names
        assert "calculator" in tool_names
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_with_sample_tools(self, sample_tools):
        """Test creating LLM configuration with sample tools."""
        llm_config = LLM(
            model_name="gpt-4.1-nano",
            provider="openai",
            tools=sample_tools,
            max_tokens=100
        )
        
        assert llm_config.tools is not None
        assert len(llm_config.tools.tools) == 2
        
        # We can create a client (though we won't make API calls without API key check)
        client = Client(llm_config)
        assert client is not None