import os
from typing import List, Optional

from anthropic import AsyncAnthropic, AsyncAnthropicVertex

from .base import BaseProvider, Message, Response


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation supporting both direct API and Vertex AI."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic client.
        
        Args:
            api_key: API key for Anthropic (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
            
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Response:
        """Create a message using Anthropic's API."""
        # Separate system messages from other messages
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                # Anthropic expects system as a separate parameter
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        if temperature is not None:
            params["temperature"] = temperature
            
        # Add any additional kwargs
        params.update(kwargs)
        
        # Make the API call
        response = await self.client.messages.create(**params)
        
        # Extract content from response
        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        
        return Response(
            content=content,
            model=response.model,
            provider=self.name,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            } if hasattr(response, 'usage') else None,
            raw_response=response
        )
    
    def validate_model_name(self, model: str) -> str:
        """Validate and transform model name for Anthropic."""
        # Common model name mappings
        model_mappings = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-opus-4": "claude-opus-4-20250514",
        }
        
        # Return mapped name if exists, otherwise return as-is
        return model_mappings.get(model, model)
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "anthropic"


class AnthropicVertexProvider(BaseProvider):
    """Anthropic provider via Google Vertex AI."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs
    ):
        """Initialize Anthropic Vertex client.
        
        Args:
            project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)
            region: GCP region (defaults to GCP_LOCATION env var)
            **kwargs: Additional configuration

        """
        super().__init__(**kwargs)
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.region = region or os.getenv("GCP_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError("project_id must be provided or GCP_PROJECT_ID must be set")
            
        self.client = AsyncAnthropicVertex(
            project_id=self.project_id,
            region=self.region
        )
    
    async def create_message(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Response:
        """Create a message using Anthropic via Vertex AI."""
        # Separate system messages from other messages
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                # Anthropic expects system as a separate parameter
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Prepare parameters
        params = {
            "model": self.validate_model_name(model),
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        if temperature is not None:
            params["temperature"] = temperature
            
        # Add any additional kwargs
        params.update(kwargs)
        
        # Make the API call
        response = await self.client.messages.create(**params)
        
        # Extract content from response
        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        
        return Response(
            content=content,
            model=response.model,
            provider=self.name,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            } if hasattr(response, 'usage') else None,
            raw_response=response
        )
    
    def validate_model_name(self, model: str) -> str:
        """Validate and transform model name for Vertex AI.
        
        Vertex AI uses @ notation instead of version dates.
        """
        # Map common names to Vertex AI format
        model_mappings = {
            "claude-3-opus-20240229": "claude-3-opus@20240229",
            "claude-3-sonnet-20240229": "claude-3-sonnet@20240229",
            "claude-3-haiku-20240307": "claude-3-haiku@20240307",
            "claude-sonnet-4-20250514": "claude-sonnet-4@20250514",
            "claude-opus-4-20250514": "claude-opus-4@20250514",
            # Short names
            "claude-3-opus": "claude-3-opus@20240229",
            "claude-3-sonnet": "claude-3-sonnet@20240229",
            "claude-3-haiku": "claude-3-haiku@20240307",
            "claude-sonnet-4": "claude-sonnet-4@20250514",
            "claude-opus-4": "claude-opus-4@20250514",
        }
        
        return model_mappings.get(model, model)
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "vertexai"
