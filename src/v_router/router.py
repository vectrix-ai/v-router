from pathlib import Path
from typing import Dict, List, Type

import yaml

from .classes.llm import LLM
from .logger import setup_logger
from .providers.anthropic import AnthropicProvider, AnthropicVertexProvider
from .providers.base import BaseProvider, Message, Response
from .providers.google import GoogleProvider, GoogleVertexProvider
from .providers.openai import AzureOpenAIProvider, OpenAIProvider

logger = setup_logger(__name__)


def load_model_config() -> Dict[str, Dict]:
    """Load model configuration from models.yml file.

    Returns:
        Dictionary with model configuration including providers and mappings

    """
    config_path = Path(__file__).parent / "models.yml"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("models", {})
    except Exception as e:
        logger.error(f"Failed to load model configuration: {e}")
        # Return empty dict if loading fails, allowing graceful degradation
        return {}


class Router:
    """Main router that handles LLM requests with automatic fallback."""

    # Provider registry mapping provider names to classes
    PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "azure": AzureOpenAIProvider,
        "google": GoogleProvider,
        "vertexai": GoogleVertexProvider,  # Can be used for both Anthropic and Google on Vertex
    }

    def __init__(self, llm_config: LLM, **provider_kwargs):
        """Initialize router with LLM configuration.

        Args:
            llm_config: Primary LLM configuration
            **provider_kwargs: Additional provider-specific configuration

        """
        self.primary_config = llm_config
        self.provider_kwargs = provider_kwargs
        self._provider_instances: Dict[str, BaseProvider] = {}
        self._model_config_cache = None

    @property
    def model_config(self) -> Dict[str, Dict]:
        """Get model configuration, loading from file if needed.

        This property allows for dynamic reloading of the configuration.
        """
        if self._model_config_cache is None:
            self._model_config_cache = load_model_config()
        return self._model_config_cache

    def reload_model_config(self) -> None:
        """Reload the model configuration from models.yml."""
        self._model_config_cache = load_model_config()
        logger.info("Reloaded model configuration from models.yml")

    def get_model_providers(self, model_name: str) -> List[str]:
        """Get available providers for a model.

        Args:
            model_name: Model name to check

        Returns:
            List of provider names that support this model

        """
        model_info = self.model_config.get(model_name, {})
        return model_info.get("providers", [])

    def get_model_mapping(self, model_name: str, provider: str) -> str:
        """Get the provider-specific model name mapping.

        Args:
            model_name: Original model name
            provider: Provider name

        Returns:
            Provider-specific model name, or original if no mapping exists

        """
        model_info = self.model_config.get(model_name, {})
        mappings = model_info.get("mappings", {})
        return mappings.get(provider, model_name)

    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get or create a provider instance.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance

        """
        if provider_name not in self._provider_instances:
            if provider_name not in self.PROVIDER_REGISTRY:
                raise ValueError(f"Unknown provider: {provider_name}")

            provider_class = self.PROVIDER_REGISTRY[provider_name]

            # Create a model mapping function for this provider
            def model_mapper(model_name: str) -> str:
                return self.get_model_mapping(model_name, provider_name)

            # Special handling for Vertex AI providers
            if provider_name == "vertexai":
                # Determine if it's Anthropic or Google based on model
                if self.primary_config.model_name.startswith("claude"):
                    self._provider_instances[provider_name] = AnthropicVertexProvider(
                        model_mapper=model_mapper, **self.provider_kwargs
                    )
                else:
                    self._provider_instances[provider_name] = GoogleVertexProvider(
                        model_mapper=model_mapper, **self.provider_kwargs
                    )
            else:
                self._provider_instances[provider_name] = provider_class(
                    model_mapper=model_mapper, **self.provider_kwargs
                )

        return self._provider_instances[provider_name]

    async def route_request(self, messages: List[Message], **kwargs) -> Response:
        """Route a request with automatic fallback handling.

        Args:
            messages: List of messages to send
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Response from the successful provider

        Raises:
            Exception: If all providers fail

        """
        errors = []

        # Try primary model first
        try:
            logger.info(
                f"Trying primary model: {self.primary_config.model_name} on {self.primary_config.provider}"
            )
            return await self._try_provider(self.primary_config, messages, **kwargs)
        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}")
            errors.append(f"Primary ({self.primary_config.provider}): {str(e)}")

        # Try backup models in priority order
        for backup_model in self.primary_config.get_ordered_backup_models():
            try:
                logger.info(
                    f"Trying backup model: {backup_model.model_name} on {backup_model.provider}"
                )
                return await self._try_provider(backup_model, messages, **kwargs)
            except Exception as e:
                logger.warning(f"Backup model failed: {str(e)}")
                errors.append(f"Backup ({backup_model.provider}): {str(e)}")

        # If try_other_providers is True, attempt same model on different providers
        if self.primary_config.try_other_providers:
            alternative_providers = self._get_alternative_providers(
                self.primary_config.model_name, exclude=[self.primary_config.provider]
            )

            for alt_provider in alternative_providers:
                try:
                    logger.info(
                        f"Trying alternative provider: {self.primary_config.model_name} on {alt_provider}"
                    )
                    alt_config = LLM(
                        model_name=self.primary_config.model_name,
                        provider=alt_provider,
                        max_tokens=self.primary_config.max_tokens,
                        temperature=self.primary_config.temperature,
                    )
                    return await self._try_provider(alt_config, messages, **kwargs)
                except Exception as e:
                    logger.warning(f"Alternative provider failed: {str(e)}")
                    errors.append(f"Alternative ({alt_provider}): {str(e)}")

        # All attempts failed
        error_summary = "\n".join(errors)
        raise Exception(f"All providers failed:\n{error_summary}")

    async def _try_provider(
        self, llm_config: LLM, messages: List[Message], **kwargs
    ) -> Response:
        """Try to send a request to a specific provider.

        Args:
            llm_config: LLM configuration to use
            messages: Messages to send
            **kwargs: Additional parameters

        Returns:
            Response from the provider

        """
        provider = self._get_provider(llm_config.provider)

        # Merge configuration parameters with kwargs
        params = {
            "model": llm_config.model_name,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return await provider.create_message(messages, **params)

    def _get_alternative_providers(
        self, model_name: str, exclude: List[str]
    ) -> List[str]:
        """Get alternative providers for a model.

        Args:
            model_name: Model name to check
            exclude: Providers to exclude

        Returns:
            List of alternative provider names

        """
        # Check exact model name first
        if model_name in self.model_config:
            providers = self.model_config[model_name].get("providers", [])
        else:
            # Try to find by prefix (e.g., "claude-3-opus" matches "claude-3-opus-20240229")
            providers = []
            for model, model_info in self.model_config.items():
                if model_name.startswith(model) or model.startswith(model_name):
                    providers.extend(model_info.get("providers", []))
            providers = list(set(providers))  # Remove duplicates

        # Filter out excluded providers
        return [p for p in providers if p not in exclude]
