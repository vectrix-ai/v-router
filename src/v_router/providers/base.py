from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..classes.tools import Tools


class Message(BaseModel):
    """Unified message format for all providers."""

    role: str  # "system", "user", "assistant"
    content: Union[
        str, List[Any], Any
    ]  # Can be string, list of content blocks, or provider-specific format
    name: Optional[str] = None


class Response(BaseModel):
    """Unified response format from all providers."""

    content: Union[
        str, List[Any], Any
    ]  # Can be string or list of content blocks for tool calls
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None  # Original provider response


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
