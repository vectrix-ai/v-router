import os
from typing import List, Optional

from google import genai

from v_router.classes.message import Message
from v_router.classes.response import Content, Response, ToolUse, Usage
from v_router.classes.tools import Tools
from v_router.providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    """Google AI Studio provider implementation."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Google AI Studio client.

        Args:
            api_key: API key for Google AI Studio (defaults to GEMINI_API_KEY env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        **kwargs,
    ) -> Response:
        """Create a message using Google's API."""
        # Convert messages to Google format
        contents = self._format_messages_for_google(messages)

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "contents": contents,
        }

        # Google uses different parameter names
        config = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if temperature is not None:
            config["temperature"] = temperature

        # Add tools if provided
        if tools:
            google_tools = self._convert_tools_to_google_format(tools)
            config["tools"] = google_tools

        if config:
            params["config"] = genai.types.GenerateContentConfig(**config)

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call (Google SDK is sync, so we'll run in executor)
        import asyncio

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.models.generate_content(**params)
        )

        # Extract content from response
        content_list = []
        tool_use_list = []

        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts

            for part in parts:
                if hasattr(part, "text") and part.text:
                    # Text content
                    content_list.append(
                        Content(type="text", role="assistant", text=part.text)
                    )
                elif hasattr(part, "function_call") and part.function_call:
                    # Function call
                    tool_use_list.append(
                        ToolUse(
                            id=f"google_{part.function_call.name}_{id(part)}",  # Google doesn't provide IDs
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args),
                        )
                    )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage_metadata.prompt_token_count
            if hasattr(response, "usage_metadata")
            else None,
            output_tokens=response.usage_metadata.candidates_token_count
            if hasattr(response, "usage_metadata")
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
            model=model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _format_messages_for_google(self, messages: List[Message]) -> List:
        """Format messages for Google API."""
        contents = []
        for msg in messages:
            if msg.role == "system":
                # Google doesn't have a separate system role, so we'll include it as user content
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[genai.types.Part(text=f"System: {msg.content}")],
                    )
                )
            elif msg.role == "user":
                contents.append(
                    genai.types.Content(
                        role="user", parts=[genai.types.Part(text=msg.content)]
                    )
                )
            elif msg.role == "assistant":
                contents.append(
                    genai.types.Content(
                        role="model",  # Google uses "model" instead of "assistant"
                        parts=[genai.types.Part(text=msg.content)],
                    )
                )
        return contents

    def _convert_tools_to_google_format(self, tools: Tools) -> List:
        """Convert Tools to Google format."""
        google_tools = []
        for tool in tools.tools:
            google_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            )
        return [genai.types.Tool(function_declarations=google_tools)]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "google"


class GoogleVertexProvider(BaseProvider):
    """Google Vertex AI provider implementation."""

    def __init__(
        self, project_id: Optional[str] = None, location: Optional[str] = None, **kwargs
    ):
        """Initialize Google Vertex AI client.

        Args:
            project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)
            location: GCP location (defaults to GCP_LOCATION env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")

        if not self.project_id:
            raise ValueError(
                "project_id must be provided or GCP_PROJECT_ID must be set"
            )

        self.client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
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
        """Create a message using Google Vertex AI."""
        # Convert messages to Google format
        contents = self._format_messages_for_google(messages)

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "contents": contents,
        }

        # Google uses different parameter names
        config = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if temperature is not None:
            config["temperature"] = temperature

        # Add tools if provided
        if tools:
            google_tools = self._convert_tools_to_google_format(tools)
            config["tools"] = google_tools

        if config:
            params["config"] = genai.types.GenerateContentConfig(**config)

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call (Google SDK is sync, so we'll run in executor)
        import asyncio

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.models.generate_content(**params)
        )

        # Extract content from response
        content_list = []
        tool_use_list = []

        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts

            for part in parts:
                if hasattr(part, "text") and part.text:
                    # Text content
                    content_list.append(
                        Content(type="text", role="assistant", text=part.text)
                    )
                elif hasattr(part, "function_call") and part.function_call:
                    # Function call
                    tool_use_list.append(
                        ToolUse(
                            id=f"google_{part.function_call.name}_{id(part)}",  # Google doesn't provide IDs
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args),
                        )
                    )

        # Build usage object
        usage = Usage(
            input_tokens=response.usage_metadata.prompt_token_count
            if hasattr(response, "usage_metadata")
            else None,
            output_tokens=response.usage_metadata.candidates_token_count
            if hasattr(response, "usage_metadata")
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
            model=model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _format_messages_for_google(self, messages: List[Message]) -> List:
        """Format messages for Google API."""
        contents = []
        for msg in messages:
            if msg.role == "system":
                # Google doesn't have a separate system role, so we'll include it as user content
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[genai.types.Part(text=f"System: {msg.content}")],
                    )
                )
            elif msg.role == "user":
                contents.append(
                    genai.types.Content(
                        role="user", parts=[genai.types.Part(text=msg.content)]
                    )
                )
            elif msg.role == "assistant":
                contents.append(
                    genai.types.Content(
                        role="model",  # Google uses "model" instead of "assistant"
                        parts=[genai.types.Part(text=msg.content)],
                    )
                )
        return contents

    def _convert_tools_to_google_format(self, tools: Tools) -> List:
        """Convert Tools to Google format."""
        google_tools = []
        for tool in tools.tools:
            google_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            )
        return [genai.types.Tool(function_declarations=google_tools)]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "vertexai"
