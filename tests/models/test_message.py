import pytest
from pydantic import ValidationError

from v_router.classes.message import Message


class TestMessage:
    """Test cases for the Message class."""
    
    def test_message_creation_valid(self):
        """Test creating a valid Message."""
        message = Message(
            role="user",
            content="Hello, how are you?"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
    
    def test_message_creation_different_roles(self):
        """Test creating messages with different roles."""
        roles_and_content = [
            ("user", "What is the weather like?"),
            ("assistant", "I can help you with that."),
            ("system", "You are a helpful assistant."),
        ]
        
        for role, content in roles_and_content:
            message = Message(role=role, content=content)
            assert message.role == role
            assert message.content == content
    
    def test_message_empty_content(self):
        """Test creating a message with empty content."""
        message = Message(
            role="user",
            content=""
        )
        
        assert message.role == "user"
        assert message.content == ""
    
    def test_message_long_content(self):
        """Test creating a message with very long content."""
        long_content = "A" * 10000
        message = Message(
            role="user",
            content=long_content
        )
        
        assert message.role == "user"
        assert message.content == long_content
        assert len(message.content) == 10000
    
    def test_message_special_characters_content(self):
        """Test creating a message with special characters."""
        special_content = "Hello! 🌍 How are you? @#$%^&*()_+-=[]{}|;':\",./<>?"
        message = Message(
            role="user",
            content=special_content
        )
        
        assert message.role == "user"
        assert message.content == special_content
    
    def test_message_multiline_content(self):
        """Test creating a message with multiline content."""
        multiline_content = """This is a multiline message.
It contains multiple lines.
And it should work perfectly fine."""
        
        message = Message(
            role="assistant",
            content=multiline_content
        )
        
        assert message.role == "assistant"
        assert message.content == multiline_content
        assert "\n" in message.content
    
    def test_message_missing_role(self):
        """Test that missing role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Message(content="Hello world")
        
        error_details = str(exc_info.value)
        assert "Field required" in error_details or "missing" in error_details.lower()
    
    def test_message_missing_content(self):
        """Test that missing content raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user")
        
        error_details = str(exc_info.value)
        assert "Field required" in error_details or "missing" in error_details.lower()
    
    def test_message_missing_both_fields(self):
        """Test that missing both fields raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Message()
        
        error_details = str(exc_info.value)
        assert "Field required" in error_details or "missing" in error_details.lower()
    
    def test_message_none_role(self):
        """Test that None role raises ValidationError."""
        with pytest.raises(ValidationError):
            Message(role=None, content="Hello")
    
    def test_message_none_content(self):
        """Test that None content raises ValidationError."""
        with pytest.raises(ValidationError):
            Message(role="user", content=None)

    def test_message_numeric_content(self):
        """Test that numeric content is converted to string."""
        message = Message(role="user", content=42)
        assert message.content == "42"
        assert isinstance(message.content, str)

class TestMessageSerialization:
    """Test serialization and deserialization of Message objects."""
    
    def test_message_dict_conversion(self):
        """Test converting Message to dict."""
        message = Message(
            role="user",
            content="Hello, world!"
        )
        
        message_dict = message.model_dump()
        
        assert isinstance(message_dict, dict)
        assert message_dict["role"] == "user"
        assert message_dict["content"] == "Hello, world!"
        assert len(message_dict) == 2
    
    def test_message_from_dict(self):
        """Test creating Message from dict."""
        message_data = {
            "role": "assistant",
            "content": "I'm here to help!"
        }
        
        message = Message.model_validate(message_data)
        
        assert message.role == "assistant"
        assert message.content == "I'm here to help!"
    
    def test_message_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original_message = Message(
            role="system",
            content="You are a helpful assistant."
        )
        
        # Serialize to JSON string
        json_str = original_message.model_dump_json()
        assert isinstance(json_str, str)
        assert "system" in json_str
        assert "helpful assistant" in json_str
        
        # Deserialize from JSON string
        reconstructed_message = Message.model_validate_json(json_str)
        
        assert reconstructed_message.role == original_message.role
        assert reconstructed_message.content == original_message.content
    
    def test_message_field_descriptions(self):
        """Test that field descriptions are preserved in schema."""
        schema = Message.model_json_schema()
        
        assert "properties" in schema
        assert "role" in schema["properties"]
        assert "content" in schema["properties"]
        
        role_field = schema["properties"]["role"]
        content_field = schema["properties"]["content"]
        
        assert "description" in role_field
        assert "description" in content_field
        assert "role of the message sender" in role_field["description"].lower()
        assert "content of the message" in content_field["description"].lower()


class TestMessageUsagePatterns:
    """Test common usage patterns for Message objects."""
    
    def test_message_conversation_list(self):
        """Test creating a list of messages representing a conversation."""
        conversation = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="The capital of France is Paris."),
            Message(role="user", content="What about Italy?"),
            Message(role="assistant", content="The capital of Italy is Rome.")
        ]
        
        assert len(conversation) == 5
        assert conversation[0].role == "system"
        assert conversation[1].role == "user"
        assert conversation[2].role == "assistant"
        
        # Test that all messages are valid
        for message in conversation:
            assert isinstance(message.role, str)
            assert isinstance(message.content, str)
            assert len(message.role) > 0
            assert len(message.content) > 0
    
    def test_message_equality(self):
        """Test message equality comparison."""
        message1 = Message(role="user", content="Hello")
        message2 = Message(role="user", content="Hello")
        message3 = Message(role="user", content="Hi")
        message4 = Message(role="assistant", content="Hello")
        
        assert message1 == message2  # Same role and content
        assert message1 != message3  # Different content
        assert message1 != message4  # Different role
    
    def test_message_copy_and_modify(self):
        """Test copying and modifying messages."""
        original = Message(role="user", content="Original message")
        
        # Create a copy with modified content
        modified = original.model_copy(update={"content": "Modified message"})
        
        assert original.content == "Original message"
        assert modified.content == "Modified message"
        assert original.role == modified.role == "user"
    
    def test_message_batch_validation(self):
        """Test validating multiple messages at once."""
        message_data_list = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        messages = [Message.model_validate(data) for data in message_data_list]
        
        assert len(messages) == 3
        assert all(isinstance(msg, Message) for msg in messages)
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
    
    def test_message_invalid_batch_validation(self):
        """Test that invalid messages in a batch are properly caught."""
        message_data_list = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},  # Missing content
            {"content": "How are you?"},  # Missing role
        ]
        
        valid_messages = []
        errors = []
        
        for data in message_data_list:
            try:
                message = Message.model_validate(data)
                valid_messages.append(message)
            except ValidationError as e:
                errors.append(e)
        
        assert len(valid_messages) == 1  # Only the first message is valid
        assert len(errors) == 2  # Two messages have validation errors
        assert valid_messages[0].role == "user"
        assert valid_messages[0].content == "Hello"
