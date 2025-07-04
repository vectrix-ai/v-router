import base64
import io
import os
from typing import Any, List, Optional, Union

import mammoth
from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

if os.getenv("LANGFUSE_HOST"):
    from langfuse import get_client
else:
    get_client = None

from v_router.classes.messages import (
    AIMessage,
    Message,
    ToolCall,
    ToolMessage,
    Usage,
)
from v_router.classes.tools import Tools
from v_router.providers.base import BaseProvider

# This will automatically emit OTEL-spans for all Anthropic API calls
AnthropicInstrumentor().instrument()

langfuse = get_client() if get_client else None


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation supporting both direct API and Vertex AI."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic client.

        Args:
            api_key: API key for Anthropic (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def create_message(
        self,
        messages: List[Union[Message, AIMessage]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> AIMessage:
        """Create a message using Anthropic's API."""
        # Separate system messages from other messages
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                # Handle AIMessage - preserve tool calls
                content = []

                # Add text content if present
                if msg.content:
                    if isinstance(msg.content, str):
                        content.append({"type": "text", "text": msg.content})
                    else:
                        # List of strings
                        for text in msg.content:
                            content.append({"type": "text", "text": text})

                # Add tool calls if present
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "input": tool_call.args,
                            }
                        )

                anthropic_messages.append({"role": "assistant", "content": content})
            elif msg.role == "system":
                # Anthropic expects system as a separate parameter
                system_message = msg.get_text_content()
            elif isinstance(msg, ToolMessage):
                # Handle tool messages - Anthropic expects tool results as user messages
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.get_text_content(),
                            }
                        ],
                    }
                )
            else:
                # Convert message content to Anthropic format
                if isinstance(msg.content, str):
                    # Simple string content
                    anthropic_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )
                else:
                    # Multimodal content
                    anthropic_content = self._convert_content_to_anthropic_format(
                        msg.content
                    )
                    anthropic_messages.append(
                        {"role": msg.role, "content": anthropic_content}
                    )

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
        }

        # Add system message if present
        if system_message:
            params["system"] = system_message

        if temperature is not None:
            params["temperature"] = temperature

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Add tool_choice if provided
            if tool_choice is not None:
                params["tool_choice"] = self._convert_tool_choice_to_anthropic_format(
                    tool_choice
                )

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.messages.create(**params)

        # Extract content from response
        content_list = []
        tool_call_list = []

        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    # Text content
                    content_list.append(content_block.text)
                elif (
                    hasattr(content_block, "type") and content_block.type == "tool_use"
                ):
                    # Tool use content
                    tool_call_list.append(
                        ToolCall(
                            id=content_block.id,
                            name=content_block.name,
                            args=content_block.input,
                        )
                    )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage.input_tokens
            if hasattr(response, "usage")
            else None,
            output_tokens=response.usage.output_tokens
            if hasattr(response, "usage")
            else None,
        )

        # Safely get raw response
        try:
            if hasattr(response, "model_dump"):
                raw_response = response.model_dump()
                if not isinstance(raw_response, dict):
                    raw_response = {}
            elif hasattr(response, "dict"):
                raw_response = response.dict()
                if not isinstance(raw_response, dict):
                    raw_response = {}
            else:
                raw_response = {}
        except Exception:
            raw_response = {}

        return AIMessage(
            content=content_list,
            tool_calls=tool_call_list,
            model=response.model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _convert_tools_to_anthropic_format(self, tools: Tools) -> list:
        """Convert Tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools.tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return anthropic_tools

    def _convert_tool_choice_to_anthropic_format(self, tool_choice):
        """Convert tool_choice to Anthropic format."""
        if isinstance(tool_choice, dict):
            # Already in provider-specific format
            return tool_choice
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "any":
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif isinstance(tool_choice, str):
            # Tool name specified
            return {"type": "tool", "name": tool_choice}
        else:
            # Default to auto
            return {"type": "auto"}

    def _convert_content_to_anthropic_format(self, content):
        """Convert message content to Anthropic format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return content

        # Handle list of content items
        if isinstance(content, list):
            anthropic_content = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        anthropic_content.append({"type": "text", "text": item.text})
                    elif item.type == "image":
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type,
                                    "data": item.data,
                                },
                            }
                        )
                    elif item.type == "document":
                        if (
                            item.media_type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        ):
                            # Convert Word document to HTML and send as text

                            try:
                                # Decode base64 data and convert to HTML
                                docx_data = base64.b64decode(item.data)
                                docx_file = io.BytesIO(docx_data)
                                result = mammoth.convert_to_html(docx_file)
                                html_content = result.value

                                anthropic_content.append(
                                    {"type": "text", "text": html_content}
                                )
                            except Exception:
                                # If conversion fails, add a placeholder message
                                anthropic_content.append(
                                    {
                                        "type": "text",
                                        "text": "[Word document provided - conversion failed]",
                                    }
                                )
                        else:
                            # Handle PDFs and other documents normally
                            anthropic_content.append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": item.media_type,
                                        "data": item.data,
                                    },
                                }
                            )
            return anthropic_content

        # Fallback to string representation
        return str(content)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "anthropic"


class AnthropicVertexProvider(BaseProvider):
    """Anthropic provider via Google Vertex AI."""

    def __init__(
        self, project_id: Optional[str] = None, region: Optional[str] = None, **kwargs
    ):
        """Initialize Anthropic Vertex client.

        Args:
            project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)
            region: GCP region (defaults to GCP_LOCATION env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.region = region or os.getenv("GCP_LOCATION", "us-central1")

        if not self.project_id:
            raise ValueError(
                "project_id must be provided or GCP_PROJECT_ID must be set"
            )

        self.client = AsyncAnthropicVertex(
            project_id=self.project_id, region=self.region
        )

    async def create_message(
        self,
        messages: List[Union[Message, AIMessage]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> AIMessage:
        """Create a message using Anthropic via Vertex AI."""
        # Separate system messages from other messages
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic expects system as a separate parameter
                system_message = msg.get_text_content()
            elif isinstance(msg, ToolMessage):
                # Handle tool messages - Anthropic expects tool results as user messages
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.get_text_content(),
                            }
                        ],
                    }
                )
            else:
                # Convert message content to Anthropic format
                if isinstance(msg.content, str):
                    # Simple string content
                    anthropic_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )
                else:
                    # Multimodal content
                    anthropic_content = self._convert_content_to_anthropic_format(
                        msg.content
                    )
                    anthropic_messages.append(
                        {"role": msg.role, "content": anthropic_content}
                    )

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
        }

        # Add system message if present
        if system_message:
            params["system"] = system_message

        if temperature is not None:
            params["temperature"] = temperature

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Add tool_choice if provided
            if tool_choice is not None:
                params["tool_choice"] = self._convert_tool_choice_to_anthropic_format(
                    tool_choice
                )

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.messages.create(**params)

        # Extract content from response
        content_list = []
        tool_call_list = []

        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    # Text content
                    content_list.append(content_block.text)
                elif (
                    hasattr(content_block, "type") and content_block.type == "tool_use"
                ):
                    # Tool use content
                    tool_call_list.append(
                        ToolCall(
                            id=content_block.id,
                            name=content_block.name,
                            args=content_block.input,
                        )
                    )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage.input_tokens
            if hasattr(response, "usage")
            else None,
            output_tokens=response.usage.output_tokens
            if hasattr(response, "usage")
            else None,
        )

        # Safely get raw response
        try:
            if hasattr(response, "model_dump"):
                raw_response = response.model_dump()
                if not isinstance(raw_response, dict):
                    raw_response = {}
            elif hasattr(response, "dict"):
                raw_response = response.dict()
                if not isinstance(raw_response, dict):
                    raw_response = {}
            else:
                raw_response = {}
        except Exception:
            raw_response = {}

        return AIMessage(
            content=content_list,
            tool_calls=tool_call_list,
            model=response.model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _convert_tools_to_anthropic_format(self, tools: Tools) -> list:
        """Convert Tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools.tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return anthropic_tools

    def _convert_tool_choice_to_anthropic_format(self, tool_choice):
        """Convert tool_choice to Anthropic format."""
        if isinstance(tool_choice, dict):
            # Already in provider-specific format
            return tool_choice
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "any":
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif isinstance(tool_choice, str):
            # Tool name specified
            return {"type": "tool", "name": tool_choice}
        else:
            # Default to auto
            return {"type": "auto"}

    def _convert_content_to_anthropic_format(self, content):
        """Convert message content to Anthropic format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return content

        # Handle list of content items
        if isinstance(content, list):
            anthropic_content = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        anthropic_content.append({"type": "text", "text": item.text})
                    elif item.type == "image":
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type,
                                    "data": item.data,
                                },
                            }
                        )
                    elif item.type == "document":
                        if (
                            item.media_type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        ):
                            # Convert Word document to HTML and send as text

                            try:
                                # Decode base64 data and convert to HTML
                                docx_data = base64.b64decode(item.data)
                                docx_file = io.BytesIO(docx_data)
                                result = mammoth.convert_to_html(docx_file)
                                html_content = result.value

                                anthropic_content.append(
                                    {"type": "text", "text": html_content}
                                )
                            except Exception:
                                # If conversion fails, add a placeholder message
                                anthropic_content.append(
                                    {
                                        "type": "text",
                                        "text": "[Word document provided - conversion failed]",
                                    }
                                )
                        else:
                            # Handle PDFs and other documents normally
                            anthropic_content.append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": item.media_type,
                                        "data": item.data,
                                    },
                                }
                            )
            return anthropic_content

        # Fallback to string representation
        return str(content)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "vertexai"
