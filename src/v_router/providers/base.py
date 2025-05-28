from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """Unified message format for all providers."""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class Response(BaseModel):
    """Unified response format from all providers."""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None  # Original provider response


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, **kwargs):
        """Initialize the provider with necessary credentials."""
        self.kwargs = kwargs
    
    @abstractmethod
    async def create_message(
        self, 
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Response:
        """Create a message using the provider's API.
        
        Args:
            messages: List of messages in the conversation
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Unified Response object
            
        """
        pass
    
    @abstractmethod
    def validate_model_name(self, model: str) -> str:
        """Validate and potentially transform model name for this provider.
        
        Args:
            model: Model name from configuration
            
        Returns:
            Provider-specific model name

        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass 