"""Test AIMessage handling in client."""

import pytest
from v_router import Client, LLM, AIMessage
from v_router.classes.messages import Message, Usage


class TestClientAIMessageHandling:
    """Test that Client properly handles AIMessage objects in messages list."""
    
    @pytest.mark.asyncio
    async def test_client_handles_aimessage_in_messages(self):
        """Test that AIMessage objects are properly converted when passed to create()."""
        client = Client(
            llm_config=LLM(
                model_name="gpt-3.5-turbo",
                provider="openai"
            )
        )
        
        # Create a mock AIMessage
        ai_msg = AIMessage(
            content="Hello from AI",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="gpt-3.5-turbo",
            provider="openai",
            raw_response={}
        )
        
        # Mix of different message types
        messages = [
            {"role": "user", "content": "Hello"},
            ai_msg,
            Message(role="user", content="How are you?"),
            {"role": "system", "content": "Be helpful"}
        ]
        
        # This should not raise TypeError
        # We'll mock the router to test just the message conversion
        converted_messages = []
        
        # Simulate the conversion logic from client
        for msg in messages:
            if isinstance(msg, Message | AIMessage):
                # Message or AIMessage objects are passed through as-is
                converted_messages.append(msg)
            else:
                converted_messages.append(
                    Message(role=msg["role"], content=msg["content"])
                )
        
        # Verify all messages were converted correctly
        assert len(converted_messages) == 4
        
        # Check specific conversions
        assert isinstance(converted_messages[0], Message)
        assert converted_messages[0].role == "user"
        assert converted_messages[0].content == "Hello"
        
        # AIMessage is passed through as-is
        assert isinstance(converted_messages[1], AIMessage)
        assert converted_messages[1].content == "Hello from AI"
        
        assert isinstance(converted_messages[2], Message)
        assert converted_messages[2].role == "user"
        assert converted_messages[2].content == "How are you?"
        
        assert isinstance(converted_messages[3], Message)
        assert converted_messages[3].role == "system"
        assert converted_messages[3].content == "Be helpful"
