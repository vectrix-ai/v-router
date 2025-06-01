import pytest

from v_router.classes.response import Response, Content, ToolUse, Usage


class TestContent:
    """Test the Content model."""
    
    def test_content_text(self):
        """Test creating text content."""
        content = Content(
            type="text",
            role="assistant",
            text="Hello, world!"
        )
        assert content.type == "text"
        assert content.role == "assistant"
        assert content.text == "Hello, world!"
    
    def test_content_tool_use(self):
        """Test creating tool use content."""
        content = Content(
            type="tool_use",
            role="assistant",
            text="Using tool..."
        )
        assert content.type == "tool_use"
        assert content.role == "assistant"
        assert content.text == "Using tool..."
    
    def test_content_validation(self):
        """Test content validation."""
        # Test invalid type
        with pytest.raises(ValueError):
            Content(
                type="invalid",
                role="assistant",
                text="Hello"
            )
        
        # Test invalid role
        with pytest.raises(ValueError):
            Content(
                type="text",
                role="invalid",
                text="Hello"
            )


class TestToolUse:
    """Test the ToolUse model."""
    
    def test_tool_use_creation(self):
        """Test creating a tool use object."""
        tool_use = ToolUse(
            id="tool_123",
            name="get_weather",
            arguments={"location": "San Francisco"}
        )
        assert tool_use.id == "tool_123"
        assert tool_use.name == "get_weather"
        assert tool_use.arguments == {"location": "San Francisco"}
    
    def test_tool_use_with_complex_arguments(self):
        """Test tool use with complex arguments."""
        tool_use = ToolUse(
            id="tool_456",
            name="search",
            arguments={
                "query": "python tutorials",
                "filters": {
                    "date": "2024",
                    "language": "en"
                },
                "limit": 10
            }
        )
        assert tool_use.id == "tool_456"
        assert tool_use.name == "search"
        assert tool_use.arguments["filters"]["date"] == "2024"
        assert tool_use.arguments["limit"] == 10


class TestUsage:
    """Test the Usage model."""
    
    def test_usage_creation(self):
        """Test creating a usage object."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
    
    def test_usage_with_none_values(self):
        """Test usage with None values."""
        usage = Usage(
            input_tokens=None,
            output_tokens=None
        )
        assert usage.input_tokens is None
        assert usage.output_tokens is None
    
    def test_usage_partial(self):
        """Test usage with partial values."""
        usage = Usage(
            input_tokens=100,
            output_tokens=None
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens is None


class TestResponse:
    """Test the Response model."""
    
    def test_response_text_only(self):
        """Test response with text content only."""
        response = Response(
            content=[
                Content(
                    type="text",
                    role="assistant",
                    text="Hello, how can I help you?"
                )
            ],
            tool_use=[],
            usage=Usage(input_tokens=10, output_tokens=8),
            model="gpt-4",
            provider="openai",
            raw_response={"test": "data"}
        )
        
        assert len(response.content) == 1
        assert response.content[0].text == "Hello, how can I help you?"
        assert len(response.tool_use) == 0
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 8
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.raw_response == {"test": "data"}
    
    def test_response_with_tool_use(self):
        """Test response with tool use."""
        response = Response(
            content=[
                Content(
                    type="text",
                    role="assistant",
                    text="I'll check the weather for you."
                )
            ],
            tool_use=[
                ToolUse(
                    id="tool_001",
                    name="get_weather",
                    arguments={"location": "New York"}
                )
            ],
            usage=Usage(input_tokens=15, output_tokens=20),
            model="claude-3-opus",
            provider="anthropic",
            raw_response={"id": "msg_123"}
        )
        
        assert len(response.content) == 1
        assert response.content[0].text == "I'll check the weather for you."
        assert len(response.tool_use) == 1
        assert response.tool_use[0].name == "get_weather"
        assert response.tool_use[0].arguments["location"] == "New York"
    
    def test_response_multiple_contents(self):
        """Test response with multiple content blocks."""
        response = Response(
            content=[
                Content(
                    type="text",
                    role="assistant",
                    text="First part of response."
                ),
                Content(
                    type="text",
                    role="assistant",
                    text="Second part of response."
                )
            ],
            tool_use=[],
            usage=Usage(input_tokens=20, output_tokens=15),
            model="gemini-pro",
            provider="google",
            raw_response={}
        )
        
        assert len(response.content) == 2
        assert response.content[0].text == "First part of response."
        assert response.content[1].text == "Second part of response."
    
    def test_response_multiple_tool_uses(self):
        """Test response with multiple tool uses."""
        response = Response(
            content=[
                Content(
                    type="text",
                    role="assistant",
                    text="I'll help you with both requests."
                )
            ],
            tool_use=[
                ToolUse(
                    id="tool_001",
                    name="get_weather",
                    arguments={"location": "Paris"}
                ),
                ToolUse(
                    id="tool_002",
                    name="get_time",
                    arguments={"timezone": "Europe/Paris"}
                )
            ],
            usage=Usage(input_tokens=25, output_tokens=30),
            model="gpt-4-turbo",
            provider="openai",
            raw_response={}
        )
        
        assert len(response.tool_use) == 2
        assert response.tool_use[0].name == "get_weather"
        assert response.tool_use[1].name == "get_time"
    
    def test_response_empty_tool_use_default(self):
        """Test that tool_use defaults to empty list."""
        response = Response(
            content=[
                Content(
                    type="text",
                    role="assistant",
                    text="Simple response"
                )
            ],
            usage=Usage(input_tokens=5, output_tokens=3),
            model="test-model",
            provider="test-provider",
            raw_response={}
        )
        
        assert response.tool_use == []