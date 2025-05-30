from abc import ABC, abstractmethod
from typing import List, Optional

from v_router.classes.message import Message
from v_router.classes.response import Response
from v_router.classes.tools import Tools


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, model_mapper=None, **kwargs):
        """Initialize the provider.

        Args:
            model_mapper: Optional function to map model names
            **kwargs: Additional provider-specific configuration

        """
        self.model_mapper = model_mapper
        self.config = kwargs

    @abstractmethod
    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Tools] = None,
        **kwargs,
    ) -> Response:
        """Create a message using the provider's API.

        Args:
            messages: List of messages to send
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            tools: Tools/functions that can be called by the model
            **kwargs: Additional provider-specific parameters

        Returns:
            Response from the provider

        """
        pass

    def validate_model_name(self, model: str) -> str:
        """Validate and transform model name for the provider.

        Args:
            model: Original model name

        Returns:
            Provider-specific model name

        """
        if self.model_mapper:
            return self.model_mapper(model)
        return model

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
