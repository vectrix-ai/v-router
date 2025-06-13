import os
from typing import Any, Dict, List

if os.getenv("LANGFUSE_HOST"):
    from langfuse import observe
else:

    def observe(**kwargs):
        """Mock observe function for testing."""

        def decorator(func):
            return func

        return decorator


from v_router.classes.llm import LLM
from v_router.classes.messages import Message
from v_router.classes.response import AIMessage
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
    async def create(
        self, messages: List[Dict[str, Any] | Message], **kwargs
    ) -> AIMessage:
        """Create a message with automatic fallback handling.

        Args:
            messages: List of message dictionaries with 'role' and 'content' or Message objects
            **kwargs: Additional parameters to pass to the provider

        Returns:
            AIMessage from the successful provider

        Example:
            response = await client.messages.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello!"}
                ]
            )

        """
        # Convert to Message objects if needed
        message_objects = []
        for msg in messages:
            if isinstance(msg, Message):
                # Already a Message object (including HumanMessage)
                message_objects.append(msg)
            else:
                # Dictionary, convert to Message
                message_objects.append(
                    Message(role=msg["role"], content=msg["content"])
                )

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
        self, messages: List[Dict[str, Any] | Message], **kwargs
    ) -> AIMessage:
        """Create a message (alternative to client.messages.create).

        Args:
            messages: List of message dictionaries or Message objects
            **kwargs: Additional parameters

        Returns:
            AIMessage from the provider

        """
        return await self.messages.create(messages, **kwargs)
