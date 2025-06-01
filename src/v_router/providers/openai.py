import json
import os
from typing import List, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from v_router.classes.message import Message
from v_router.classes.response import Content, Response, ToolUse, Usage
from v_router.classes.tools import Tools
from v_router.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI client.

        Args:
            api_key: API key for OpenAI (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        **kwargs,
    ) -> Response:
        """Create a message using OpenAI's API."""
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                # Simple string content
                openai_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Multimodal content
                openai_content = self._convert_content_to_openai_format(msg.content)
                openai_messages.append({"role": msg.role, "content": openai_content})

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": openai_messages,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_openai_format(tools)

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.chat.completions.create(**params)

        # Extract content from response
        content_list = []
        tool_use_list = []

        if response.choices and response.choices[0].message:
            message = response.choices[0].message

            # Add text content if present
            if message.content:
                content_list.append(
                    Content(type="text", role="assistant", text=message.content)
                )

            # Check if there are tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function:
                        tool_use_list.append(
                            ToolUse(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=json.loads(
                                    tool_call.function.arguments
                                ),  # OpenAI returns as JSON string
                            )
                        )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage.prompt_tokens
            if hasattr(response, "usage")
            else None,
            output_tokens=response.usage.completion_tokens
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

        return Response(
            content=content_list,
            tool_use=tool_use_list,
            model=response.model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _convert_tools_to_openai_format(self, tools: Tools) -> list:
        """Convert Tools to OpenAI format."""
        openai_tools = []
        for tool in tools.tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return openai_tools

    def _convert_content_to_openai_format(self, content):
        """Convert message content to OpenAI format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return content

        # Handle list of content items
        if isinstance(content, list):
            openai_content = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        openai_content.append({"type": "text", "text": item.text})
                    elif item.type == "image":
                        # OpenAI expects base64 images with data URI scheme
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.media_type};base64,{item.data}"
                                },
                            }
                        )
                    elif item.type == "document":
                        # OpenAI doesn't natively support PDFs, so we'll add a text note
                        openai_content.append(
                            {
                                "type": "text",
                                "text": "[PDF document provided - OpenAI does not support PDF viewing]",
                            }
                        )
            return openai_content

        # Fallback to string representation
        return str(content)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "openai"


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Azure OpenAI client.

        Args:
            api_key: API key for Azure (defaults to AZURE_OPENAI_API_KEY env var)
            azure_endpoint: Azure endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: API version (defaults to "2025-01-01-preview")
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or "2025-01-01-preview"

        if not self.azure_endpoint:
            raise ValueError(
                "azure_endpoint must be provided or AZURE_OPENAI_ENDPOINT must be set"
            )

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        **kwargs,
    ) -> Response:
        """Create a message using Azure OpenAI's API.

        Note: In Azure, the model parameter refers to your deployment name,
        not the actual model name.
        """
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                # Simple string content
                openai_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Multimodal content
                openai_content = self._convert_content_to_openai_format(msg.content)
                openai_messages.append({"role": msg.role, "content": openai_content})

        # Prepare parameters
        params = {
            "model": model,  # This should be the deployment name in Azure
            "messages": openai_messages,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_openai_format(tools)

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.chat.completions.create(**params)

        # Extract content from response
        content_list = []
        tool_use_list = []

        if response.choices and response.choices[0].message:
            message = response.choices[0].message

            # Add text content if present
            if message.content:
                content_list.append(
                    Content(type="text", role="assistant", text=message.content)
                )

            # Check if there are tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function:
                        tool_use_list.append(
                            ToolUse(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=json.loads(
                                    tool_call.function.arguments
                                ),  # OpenAI returns as JSON string
                            )
                        )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage.prompt_tokens
            if hasattr(response, "usage")
            else None,
            output_tokens=response.usage.completion_tokens
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

        return Response(
            content=content_list,
            tool_use=tool_use_list,
            model=response.model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _convert_tools_to_openai_format(self, tools: Tools) -> list:
        """Convert Tools to OpenAI format."""
        openai_tools = []
        for tool in tools.tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return openai_tools

    def _convert_content_to_openai_format(self, content):
        """Convert message content to OpenAI format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return content

        # Handle list of content items
        if isinstance(content, list):
            openai_content = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        openai_content.append({"type": "text", "text": item.text})
                    elif item.type == "image":
                        # OpenAI expects base64 images with data URI scheme
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.media_type};base64,{item.data}"
                                },
                            }
                        )
                    elif item.type == "document":
                        # OpenAI doesn't natively support PDFs, so we'll add a text note
                        openai_content.append(
                            {
                                "type": "text",
                                "text": "[PDF document provided - OpenAI does not support PDF viewing]",
                            }
                        )
            return openai_content

        # Fallback to string representation
        return str(content)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "azure"
