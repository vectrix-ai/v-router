"""Tests for multimodal content handling in v-router."""

import base64
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from v_router.classes.messages import (
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
)
from v_router.providers.anthropic import AnthropicProvider
from v_router.providers.google import GoogleProvider
from v_router.providers.openai import OpenAIProvider


class TestMultimodalContent:
    """Test multimodal content handling."""

    def test_message_with_text_content(self):
        """Test creating a message with text content."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.get_text_content() == "Hello, world!"

    def test_message_with_multimodal_content(self):
        """Test creating a message with multimodal content."""
        content = [
            TextContent(text="What's in this image?"),
            ImageContent(data="base64data", media_type="image/png"),
        ]
        msg = Message(role="user", content=content)
        assert msg.role == "user"
        assert len(msg.content) == 2
        assert msg.content[0].text == "What's in this image?"
        assert msg.content[1].data == "base64data"
        assert msg.get_text_content() == "What's in this image? [image]"

    def test_message_with_file_path(self, tmp_path):
        """Test that file paths are converted to multimodal content."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        # Write a minimal PNG header
        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        )

        msg = Message(role="user", content=str(image_path))
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ImageContent)
        assert msg.content[0].media_type == "image/png"

    def test_message_with_pdf_path(self, tmp_path):
        """Test that PDF paths are converted to document content."""
        # Create a test PDF file
        pdf_path = tmp_path / "test.pdf"
        # Write a minimal PDF header
        pdf_path.write_bytes(b"%PDF-1.4")

        msg = Message(role="user", content=str(pdf_path))
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], DocumentContent)
        assert msg.content[0].media_type == "application/pdf"

    def test_message_with_unsupported_image_type(self, tmp_path):
        """Test that unsupported image types raise an error."""
        # Create a test TIFF file (TIFF is detected but not supported)
        tiff_path = tmp_path / "test.tiff"
        # Write a minimal TIFF header
        tiff_path.write_bytes(b"II*\x00\x08\x00\x00\x00")

        with pytest.raises(ValueError, match="Unsupported image type: image/tiff"):
            Message(role="user", content=str(tiff_path))

    def test_message_numeric_content(self):
        """Test that numeric content is converted to string."""
        msg = Message(role="user", content=42)
        assert msg.content == "42"

        msg = Message(role="user", content=3.14)
        assert msg.content == "3.14"

    def test_message_long_string_not_treated_as_path(self):
        """Test that long strings are not checked as file paths."""
        long_content = "A" * 1000
        msg = Message(role="user", content=long_content)
        assert msg.content == long_content

    def test_message_string_without_path_separator(self):
        """Test that strings without path separators are not checked as paths."""
        content = "This is just a regular message with no slashes"
        msg = Message(role="user", content=content)
        assert msg.content == content


class TestProviderMultimodalHandling:
    """Test how providers handle multimodal content."""

    @pytest.mark.asyncio
    async def test_anthropic_multimodal_conversion(self):
        """Test Anthropic provider converts multimodal content correctly."""
        provider = AnthropicProvider()

        # Mock the client
        mock_response = Mock()
        mock_response.content = [Mock(text="I see an image", type="text")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3"

        with patch.object(
            provider.client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [
                Message(
                    role="user",
                    content=[
                        TextContent(text="What's this?"),
                        ImageContent(data="base64data", media_type="image/jpeg"),
                    ],
                )
            ]

            await provider.create_message(messages, "claude-3", max_tokens=100)

            # Check that the multimodal content was converted correctly
            call_args = mock_create.call_args[1]
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert isinstance(call_args["messages"][0]["content"], list)
            assert len(call_args["messages"][0]["content"]) == 2
            assert call_args["messages"][0]["content"][0] == {
                "type": "text",
                "text": "What's this?",
            }
            assert call_args["messages"][0]["content"][1] == {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "base64data",
                },
            }

    @pytest.mark.asyncio
    async def test_google_multimodal_conversion(self):
        """Test Google provider converts multimodal content correctly."""
        provider = GoogleProvider()

        # Mock the response
        mock_response = Mock()
        mock_response.candidates = [
            Mock(
                content=Mock(
                    parts=[Mock(text="I see an image", function_call=None)]
                )
            )
        ]
        mock_response.usage_metadata = Mock(
            prompt_token_count=10, candidates_token_count=5
        )

        with patch.object(
            provider.client.models,
            "generate_content",
            return_value=mock_response,
        ) as mock_generate:
            messages = [
                Message(
                    role="user",
                    content=[
                        TextContent(text="What's this?"),
                        ImageContent(data="YmFzZTY0ZGF0YQ==", media_type="image/jpeg"),
                    ],
                )
            ]

            await provider.create_message(messages, "gemini-1.5-flash", max_tokens=100)

            # Check that the multimodal content was converted correctly
            call_args = mock_generate.call_args[1]
            assert len(call_args["contents"]) == 1
            assert call_args["contents"][0].role == "user"
            assert len(call_args["contents"][0].parts) == 2
            # First part should be text
            assert hasattr(call_args["contents"][0].parts[0], "text")
            # Second part should be inline data
            assert hasattr(call_args["contents"][0].parts[1], "inline_data")

    @pytest.mark.asyncio
    async def test_openai_multimodal_conversion(self):
        """Test OpenAI provider converts multimodal content correctly."""
        provider = OpenAIProvider()

        # Mock the response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="I see an image", tool_calls=None))
        ]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "gpt-4o"

        with patch.object(
            provider.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [
                Message(
                    role="user",
                    content=[
                        TextContent(text="What's this?"),
                        ImageContent(data="base64data", media_type="image/jpeg"),
                    ],
                )
            ]

            await provider.create_message(messages, "gpt-4o", max_tokens=100)

            # Check that the multimodal content was converted correctly
            call_args = mock_create.call_args[1]
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert isinstance(call_args["messages"][0]["content"], list)
            assert len(call_args["messages"][0]["content"]) == 2
            assert call_args["messages"][0]["content"][0] == {
                "type": "text",
                "text": "What's this?",
            }
            assert call_args["messages"][0]["content"][1] == {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,base64data"},
            }

    @pytest.mark.asyncio
    async def test_openai_pdf_handling(self):
        """Test that OpenAI handles PDFs using the responses API."""
        provider = OpenAIProvider()

        # Mock the responses API response
        mock_response = Mock()
        mock_response.output = [
            Mock(content=[Mock(text="I can see the PDF content")])
        ]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "gpt-4o"

        with patch.object(
            provider.client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            # Use valid base64 data
            import base64
            pdf_data = base64.b64encode(b"fake pdf content").decode("utf-8")
            
            messages = [
                Message(
                    role="user",
                    content=[
                        TextContent(text="Read this PDF"),
                        DocumentContent(data=pdf_data),
                    ],
                )
            ]

            response = await provider.create_message(messages, "gpt-4o", max_tokens=100)

            # Check that responses API was used and PDF was sent via input_file
            call_args = mock_create.call_args[1]
            assert "input" in call_args
            assert len(call_args["input"][0]["content"]) == 2
            assert call_args["input"][0]["content"][1]["type"] == "input_file"
            assert call_args["input"][0]["content"][1]["filename"] == "document.pdf"
            
            # Check the response content
            assert response.content == ["I can see the PDF content"]

    @pytest.mark.asyncio
    async def test_backward_compatibility_string_content(self):
        """Test that string content still works for backward compatibility."""
        providers = [
            AnthropicProvider(),
            GoogleProvider(),
            OpenAIProvider(),
        ]

        for provider in providers:
            # Mock appropriate response based on provider
            if isinstance(provider, AnthropicProvider):
                mock_response = Mock()
                mock_response.content = [Mock(text="Hello", type="text")]
                mock_response.usage = Mock(input_tokens=10, output_tokens=5)
                mock_response.model = "claude-3"
                mock_method = provider.client.messages.create
            elif isinstance(provider, GoogleProvider):
                mock_response = Mock()
                mock_response.candidates = [
                    Mock(
                        content=Mock(parts=[Mock(text="Hello", function_call=None)])
                    )
                ]
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5
                )
                mock_method = provider.client.models.generate_content
            else:  # OpenAI
                mock_response = Mock()
                mock_response.choices = [
                    Mock(message=Mock(content="Hello", tool_calls=None))
                ]
                mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
                mock_response.model = "gpt-4"
                mock_method = provider.client.chat.completions.create

            with patch.object(
                mock_method.__self__,
                mock_method.__name__,
                new_callable=AsyncMock if isinstance(provider, (AnthropicProvider, OpenAIProvider)) else Mock,
                return_value=mock_response,
            ):
                messages = [Message(role="user", content="Hello, world!")]
                response = await provider.create_message(
                    messages, "test-model", max_tokens=100
                )
                if isinstance(response.content, list):
                    assert response.content[0] == "Hello"
                else:
                    assert response.content == "Hello"