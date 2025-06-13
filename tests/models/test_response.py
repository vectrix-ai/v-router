import pytest

from v_router.classes.messages import AIMessage, ToolCall, Usage


# Content class tests removed as Content class no longer exists


class TestToolCall:
    """Test the ToolCall model."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call object."""
        tool_call = ToolCall(
            id="tool_123",
            name="get_weather",
            args={"location": "San Francisco"}
        )
        assert tool_call.id == "tool_123"
        assert tool_call.name == "get_weather"
        assert tool_call.args == {"location": "San Francisco"}
    
    def test_tool_call_with_complex_args(self):
        """Test tool call with complex arguments."""
        tool_call = ToolCall(
            id="tool_456",
            name="search",
            args={
                "query": "python tutorials",
                "filters": {
                    "date": "2024",
                    "language": "en"
                },
                "limit": 10
            }
        )
        assert tool_call.id == "tool_456"
        assert tool_call.name == "search"
        assert tool_call.args["filters"]["date"] == "2024"
        assert tool_call.args["limit"] == 10


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


class TestAIMessage:
    """Test the AIMessage model."""
    
    def test_response_text_only(self):
        """Test response with text content only."""
        response = AIMessage(
            content="Hello, how can I help you?",
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=8),
            model="gpt-4.1-nano",
            provider="openai",
            raw_response={"test": "data"}
        )
        
        assert response.content == "Hello, how can I help you?"
        assert len(response.tool_calls) == 0
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 8
        assert response.model == "gpt-4.1-nano"
        assert response.provider == "openai"
        assert response.raw_response == {"test": "data"}
    
    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        response = AIMessage(
            content="I'll check the weather for you.",
            tool_calls=[
                ToolCall(
                    id="tool_001",
                    name="get_weather",
                    args={"location": "New York"}
                )
            ],
            usage=Usage(input_tokens=15, output_tokens=20),
            model="claude-3-opus",
            provider="anthropic",
            raw_response={"id": "msg_123"}
        )
        
        assert response.content == "I'll check the weather for you."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].args["location"] == "New York"
    
    def test_response_multiple_contents(self):
        """Test response with multiple content blocks."""
        response = AIMessage(
            content=["First part of response.", "Second part of response."],
            tool_calls=[],
            usage=Usage(input_tokens=20, output_tokens=15),
            model="gemini-pro",
            provider="google",
            raw_response={}
        )
        
        assert len(response.content) == 2
        assert response.content[0] == "First part of response."
        assert response.content[1] == "Second part of response."
    
    def test_response_multiple_tool_calls(self):
        """Test response with multiple tool calls."""
        response = AIMessage(
            content="I'll help you with both requests.",
            tool_calls=[
                ToolCall(
                    id="tool_001",
                    name="get_weather",
                    args={"location": "Paris"}
                ),
                ToolCall(
                    id="tool_002",
                    name="get_time",
                    args={"timezone": "Europe/Paris"}
                )
            ],
            usage=Usage(input_tokens=25, output_tokens=30),
            model="gpt-4.1-nano",
            provider="openai",
            raw_response={}
        )
        
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[1].name == "get_time"
    
    def test_response_empty_tool_calls_default(self):
        """Test that tool_calls defaults to empty list."""
        response = AIMessage(
            content="Simple response",
            usage=Usage(input_tokens=5, output_tokens=3),
            model="test-model",
            provider="test-provider",
            raw_response={}
        )
        
        assert response.tool_calls == []
    
    def test_get_text_content_string(self):
        """Test get_text_content with string content."""
        response = AIMessage(
            content="This is a simple text response",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
            provider="test-provider",
            raw_response={}
        )
        
        assert response.get_text_content() == "This is a simple text response"
    
    def test_get_text_content_list(self):
        """Test get_text_content with list content."""
        response = AIMessage(
            content=["Hello", "world", "from", "AI"],
            usage=Usage(input_tokens=10, output_tokens=4),
            model="test-model",
            provider="test-provider",
            raw_response={}
        )
        
        assert response.get_text_content() == "Hello world from AI"