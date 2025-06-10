from typing import Any, Dict, List

from langfuse import observe

from v_router.classes.llm import LLM
from v_router.providers.base import Message, Response
from v_router.router import Router


class Messages:
    """Messages API interface."""

    def __init__(self, router: Router):
        """Initialize Messages API.

        Args:
            router: Router instance to handle requests

        """
        self.router = router

    @observe(name="v-router-call", as_type="generation")
    async def create(self, messages: List[Dict[str, Any]], **kwargs) -> Response:
        """Create a message with automatic fallback handling.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Response from the successful provider

        Example:
            response = await client.messages.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello!"}
                ]
            )

        """
        # Convert dict messages to Message objects
        message_objects = [
            Message(role=msg["role"], content=msg["content"]) for msg in messages
        ]

        return await self.router.route_request(message_objects, **kwargs)


class Client:
    """Main client for v-router.

    Provides a unified interface for multiple LLM providers with automatic fallback.
    """

    def __init__(self, llm_config: LLM, **provider_kwargs):
        """Initialize the v-router client.

        Args:
            llm_config: LLM configuration with primary and backup models
            **provider_kwargs: Additional provider-specific configuration
                - api_key: API key for the provider
                - project_id: GCP project ID (for Vertex AI)
                - region/location: Region for cloud providers
                - azure_endpoint: Azure OpenAI endpoint
                - api_version: API version for Azure

        Example:
            from v_router import Client, LLM, BackupModel

            llm_config = LLM(
                model_name="claude-3-opus",
                provider="anthropic",
                max_tokens=1000,
                temperature=0.7,
                backup_models=[
                    BackupModel(
                        model=LLM(
                            model_name="gpt-4",
                            provider="openai"
                        ),
                        priority=1
                    )
                ],
                try_other_providers=True
            )

            client = Client(llm_config)

        """
        self.router = Router(llm_config, **provider_kwargs)
        self.messages = Messages(self.router)

    async def create_message(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Response:
        """Create a message (alternative to client.messages.create).

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Response from the provider

        """
        return await self.messages.create(messages, **kwargs)
