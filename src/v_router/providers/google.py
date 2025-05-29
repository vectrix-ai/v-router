import os
from typing import List, Optional

from google import genai

from .base import BaseProvider, Message, Response


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
        **kwargs,
    ) -> Response:
        """Create a message using Google's API."""
        # Convert messages to Google format
        # Google uses a different format - combine all messages into a single prompt
        prompt = self._format_messages_as_prompt(messages)

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "contents": prompt,
        }

        # Google uses different parameter names
        config = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if temperature is not None:
            config["temperature"] = temperature

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
        content = ""
        if response.candidates and response.candidates[0].content.parts:
            content = response.candidates[0].content.parts[0].text

        return Response(
            content=content,
            model=model,
            provider=self.name,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
            if hasattr(response, "usage_metadata")
            else None,
            raw_response=response,
        )

    def _format_messages_as_prompt(self, messages: List[Message]) -> str:
        """Format messages into a single prompt for Google."""
        formatted_parts = []
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")

        return "\n\n".join(formatted_parts)

    def validate_model_name(self, model: str) -> str:
        """Validate and transform model name for Google."""
        # Common model name mappings
        model_mappings = {
            "gemini-pro": "gemini-1.0-pro",
            "gemini-pro-vision": "gemini-1.0-pro-vision",
            "gemini-1.5-pro": "gemini-1.5-pro-latest",
            "gemini-1.5-flash": "gemini-1.5-flash-latest",
            "gemini-2-flash": "gemini-2.0-flash-001",
            "gemini-2.0-flash": "gemini-2.0-flash-001",
        }

        # Return mapped name if exists, otherwise return as-is
        return model_mappings.get(model, model)

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
        **kwargs,
    ) -> Response:
        """Create a message using Google Vertex AI."""
        # Convert messages to Google format
        prompt = self._format_messages_as_prompt(messages)

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "contents": prompt,
        }

        # Google uses different parameter names
        config = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if temperature is not None:
            config["temperature"] = temperature

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
        content = ""
        if response.candidates and response.candidates[0].content.parts:
            content = response.candidates[0].content.parts[0].text

        return Response(
            content=content,
            model=model,
            provider=self.name,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
            if hasattr(response, "usage_metadata")
            else None,
            raw_response=response,
        )

    def _format_messages_as_prompt(self, messages: List[Message]) -> str:
        """Format messages into a single prompt for Google."""
        formatted_parts = []
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")

        return "\n\n".join(formatted_parts)

    def validate_model_name(self, model: str) -> str:
        """Validate and transform model name for Vertex AI."""
        # Common model name mappings (same as regular Google)
        model_mappings = {
            "gemini-pro": "gemini-1.0-pro",
            "gemini-pro-vision": "gemini-1.0-pro-vision",
            "gemini-1.5-pro": "gemini-1.5-pro-latest",
            "gemini-1.5-flash": "gemini-1.5-flash-latest",
            "gemini-2-flash": "gemini-2.0-flash-001",
            "gemini-2.0-flash": "gemini-2.0-flash-001",
        }

        # Return mapped name if exists, otherwise return as-is
        return model_mappings.get(model, model)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "vertexai"
