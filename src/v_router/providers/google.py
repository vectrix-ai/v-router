import base64
import io
import os
from typing import Any, List, Optional

import mammoth
from google import genai

if os.getenv("LANGFUSE_HOST"):
    from langfuse import get_client
else:
    get_client = None

from v_router.classes.messages import Message, ToolMessage
from v_router.classes.response import AIMessage, Content, ToolCall, Usage
from v_router.classes.tools import Tools
from v_router.providers.base import BaseProvider

langfuse = get_client() if get_client else None


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
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> AIMessage:
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

            # Add tool_choice if provided
            if tool_choice is not None:
                tool_config = self._convert_tool_choice_to_google_format(tool_choice)
                config["tool_config"] = tool_config

        if config:
            params["config"] = genai.types.GenerateContentConfig(**config)

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call (Google SDK is sync, so we'll run in executor)
        import asyncio

        if langfuse:
            with langfuse.start_as_current_generation(
                name="GoogleGenerativeAI",
                model=model,
                input=messages,
                model_parameters={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tools.model_dump() if tools else None,
                    "tool_choice": tool_choice,
                },
            ) as generation:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.models.generate_content(**params)
                )

                # Extract content from response
                content_list = []
                tool_call_list = []

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
                            tool_call_list.append(
                                ToolCall(
                                    id=f"google_{part.function_call.name}_{id(part)}",  # Google doesn't provide IDs
                                    name=part.function_call.name,
                                    args=dict(part.function_call.args),
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

                generation.update(
                    output=content_list,
                    usage_details={
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": response.usage_metadata.total_token_count,
                    },
                )

                return AIMessage(
                    content=content_list,
                    tool_calls=tool_call_list,
                    model=model,
                    provider=self.name,
                    usage=usage,
                    raw_response=raw_response,
                )
        else:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.models.generate_content(**params)
            )

            # Extract content from response
            content_list = []
            tool_call_list = []

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
                        tool_call_list.append(
                            ToolCall(
                                id=f"google_{part.function_call.name}_{id(part)}",  # Google doesn't provide IDs
                                name=part.function_call.name,
                                args=dict(part.function_call.args),
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

            return AIMessage(
                content=content_list,
                tool_calls=tool_call_list,
                model=model,
                provider=self.name,
                usage=usage,
                raw_response=raw_response,
            )

    def _format_messages_for_google(self, messages: List[Message]) -> List:
        """Format messages for Google API."""
        contents = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Handle tool messages - Google expects function responses as user messages
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part.from_function_response(
                                name=msg.tool_call_id.split("_")[1]
                                if "_" in msg.tool_call_id
                                else msg.tool_call_id,
                                response={"result": msg.get_text_content()},
                            )
                        ],
                    )
                )
            elif msg.role == "system":
                # Google doesn't have a separate system role, so we'll include it as user content
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part(text=f"System: {msg.get_text_content()}")
                        ],
                    )
                )
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    contents.append(
                        genai.types.Content(
                            role="user", parts=[genai.types.Part(text=msg.content)]
                        )
                    )
                else:
                    parts = self._convert_content_to_google_parts(msg.content)
                    contents.append(genai.types.Content(role="user", parts=parts))
            elif msg.role == "assistant":
                if isinstance(msg.content, str):
                    contents.append(
                        genai.types.Content(
                            role="model",  # Google uses "model" instead of "assistant"
                            parts=[genai.types.Part(text=msg.content)],
                        )
                    )
                else:
                    parts = self._convert_content_to_google_parts(msg.content)
                    contents.append(
                        genai.types.Content(
                            role="model",  # Google uses "model" instead of "assistant"
                            parts=parts,
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

    def _convert_tool_choice_to_google_format(self, tool_choice):
        """Convert tool_choice to Google format."""
        if isinstance(tool_choice, dict):
            # Already in provider-specific format
            return genai.types.ToolConfig(**tool_choice)
        elif tool_choice == "auto":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="AUTO")
            )
        elif tool_choice == "any":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="ANY")
            )
        elif tool_choice == "none":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="NONE")
            )
        elif isinstance(tool_choice, str):
            # Tool name specified
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool_choice]
                )
            )
        else:
            # Default to auto
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="AUTO")
            )

    def _convert_content_to_google_parts(self, content) -> List:
        """Convert message content to Google Part format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return [genai.types.Part(text=content)]

        # Handle list of content items
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        parts.append(genai.types.Part(text=item.text))
                    elif item.type == "image":
                        # Decode base64 string to bytes
                        parts.append(
                            genai.types.Part(
                                inline_data=genai.types.Blob(
                                    mime_type=item.media_type,
                                    data=base64.b64decode(item.data),
                                )
                            )
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

                                parts.append(genai.types.Part(text=html_content))
                            except Exception:
                                # If conversion fails, add a placeholder message
                                parts.append(
                                    genai.types.Part(
                                        text="[Word document provided - conversion failed]"
                                    )
                                )
                        else:
                            # Google Gemini supports PDF files through inline_data
                            parts.append(
                                genai.types.Part(
                                    inline_data=genai.types.Blob(
                                        mime_type=item.media_type,
                                        data=base64.b64decode(item.data),
                                    )
                                )
                            )
            return parts

        # Fallback to text
        return [genai.types.Part(text=str(content))]

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
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> AIMessage:
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

            # Add tool_choice if provided
            if tool_choice is not None:
                tool_config = self._convert_tool_choice_to_google_format(tool_choice)
                config["tool_config"] = tool_config

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
        tool_call_list = []

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
                    tool_call_list.append(
                        ToolCall(
                            id=f"google_{part.function_call.name}_{id(part)}",  # Google doesn't provide IDs
                            name=part.function_call.name,
                            args=dict(part.function_call.args),
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

        return AIMessage(
            content=content_list,
            tool_calls=tool_call_list,
            model=model,
            provider=self.name,
            usage=usage,
            raw_response=raw_response,
        )

    def _format_messages_for_google(self, messages: List[Message]) -> List:
        """Format messages for Google API."""
        contents = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Handle tool messages - Google expects function responses as user messages
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part.from_function_response(
                                name=msg.tool_call_id.split("_")[1]
                                if "_" in msg.tool_call_id
                                else msg.tool_call_id,
                                response={"result": msg.get_text_content()},
                            )
                        ],
                    )
                )
            elif msg.role == "system":
                # Google doesn't have a separate system role, so we'll include it as user content
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part(text=f"System: {msg.get_text_content()}")
                        ],
                    )
                )
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    contents.append(
                        genai.types.Content(
                            role="user", parts=[genai.types.Part(text=msg.content)]
                        )
                    )
                else:
                    parts = self._convert_content_to_google_parts(msg.content)
                    contents.append(genai.types.Content(role="user", parts=parts))
            elif msg.role == "assistant":
                if isinstance(msg.content, str):
                    contents.append(
                        genai.types.Content(
                            role="model",  # Google uses "model" instead of "assistant"
                            parts=[genai.types.Part(text=msg.content)],
                        )
                    )
                else:
                    parts = self._convert_content_to_google_parts(msg.content)
                    contents.append(
                        genai.types.Content(
                            role="model",  # Google uses "model" instead of "assistant"
                            parts=parts,
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

    def _convert_tool_choice_to_google_format(self, tool_choice):
        """Convert tool_choice to Google format."""
        if isinstance(tool_choice, dict):
            # Already in provider-specific format
            return genai.types.ToolConfig(**tool_choice)
        elif tool_choice == "auto":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="AUTO")
            )
        elif tool_choice == "any":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="ANY")
            )
        elif tool_choice == "none":
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="NONE")
            )
        elif isinstance(tool_choice, str):
            # Tool name specified
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool_choice]
                )
            )
        else:
            # Default to auto
            return genai.types.ToolConfig(
                function_calling_config=genai.types.FunctionCallingConfig(mode="AUTO")
            )

    def _convert_content_to_google_parts(self, content) -> List:
        """Convert message content to Google Part format."""
        # Handle string content (backward compatibility)
        if isinstance(content, str):
            return [genai.types.Part(text=content)]

        # Handle list of content items
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        parts.append(genai.types.Part(text=item.text))
                    elif item.type == "image":
                        # Decode base64 string to bytes
                        parts.append(
                            genai.types.Part(
                                inline_data=genai.types.Blob(
                                    mime_type=item.media_type,
                                    data=base64.b64decode(item.data),
                                )
                            )
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

                                parts.append(genai.types.Part(text=html_content))
                            except Exception:
                                # If conversion fails, add a placeholder message
                                parts.append(
                                    genai.types.Part(
                                        text="[Word document provided - conversion failed]"
                                    )
                                )
                        else:
                            # Google Gemini supports PDF files through inline_data
                            parts.append(
                                genai.types.Part(
                                    inline_data=genai.types.Blob(
                                        mime_type=item.media_type,
                                        data=base64.b64decode(item.data),
                                    )
                                )
                            )
            return parts

        # Fallback to text
        return [genai.types.Part(text=str(content))]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "vertexai"
