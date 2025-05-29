import os
from typing import List, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from v_router.providers.base import BaseProvider, Message, Response


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
        **kwargs,
    ) -> Response:
        """Create a message using OpenAI's API."""
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": openai_messages,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.chat.completions.create(**params)

        # Extract content from response
        content = response.choices[0].message.content if response.choices else ""

        return Response(
            content=content,
            model=response.model,
            provider=self.name,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if hasattr(response, "usage")
            else None,
            raw_response=response,
        )

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
        **kwargs,
    ) -> Response:
        """Create a message using Azure OpenAI's API.

        Note: In Azure, the model parameter refers to your deployment name,
        not the actual model name.
        """
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Prepare parameters
        params = {
            "model": model,  # This should be the deployment name in Azure
            "messages": openai_messages,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self.client.chat.completions.create(**params)

        # Extract content from response
        content = response.choices[0].message.content if response.choices else ""

        return Response(
            content=content,
            model=response.model,
            provider=self.name,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if hasattr(response, "usage")
            else None,
            raw_response=response,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "azure"
