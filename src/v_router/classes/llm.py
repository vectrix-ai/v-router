from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .tools import ToolCall, Tools


class BackupModel(BaseModel):
    """Configuration for a backup LLM model.

    This model is used when the primary LLM fails or is unavailable.
    It contains the model configuration and its priority.
    """

    model: LLM = Field(..., description="The backup LLM model configuration.")
    priority: int = Field(
        ...,
        description="Priority of this backup model. Lower numbers = higher priority (1 is highest priority).",
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        """Validate the priority value."""
        if v < 1:
            raise ValueError("Priority must be at least 1")
        return v


class LLM(BaseModel):
    """Configuration for a Large Language Model (LLM).

    This model defines the parameters for using an LLM, including the model name provider, and optional parameters like max tokens and temperature.

    """

    model_name: str = Field(
        ...,
        description="Name of the LLM model to be used. For example, 'gpt-4.1' or 'claude-sonnet-4'.",
    )
    provider: Literal["openai", "anthropic", "azure", "google", "vertexai"] = Field(
        ...,
        description="Provider of the LLM service. Can be 'openai', 'anthropic', 'azure', 'google', or 'vertexai'.",
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate in the response. If not specified, defaults to model's maximum.",
    )
    temperature: Optional[float] = Field(
        0,
        description="Sampling temperature for the model. Higher values (up to 1.0) make output more random, lower values make it more deterministic.",
    )
    backup_models: List[BackupModel] = Field(
        default_factory=list,
        description="List of backup models with priorities. Models will be tried in order of priority (lowest number first).",
    )
    try_other_providers: bool = Field(
        False,
        description="Try other providers for this model as backup (if available). Only the exact same model will be used for this.",
    )
    tools: Optional[Tools] = Field(
        None,
        description="Tools/functions that can be called by the model. Can be a Tools object or a list of ToolCall objects.",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description=(
            "Controls how the model uses tools. Can be:\n"
            "- None or 'auto': Model decides whether to use tools (default)\n"
            "- 'any': Model must use one of the provided tools\n"
            "- 'none': Model is prevented from using tools\n"
            "- str (tool name): Force the model to use a specific tool\n"
            "- dict: Provider-specific format for advanced control"
        ),
    )

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, v):
        """Validate and normalize tools parameter."""
        if v is None:
            return None
        
        # If it's already a Tools object, return as-is
        if isinstance(v, Tools):
            return v
        
        # If it's a list of ToolCall objects, wrap in Tools
        if isinstance(v, list):
            # Validate that all items are ToolCall instances
            for item in v:
                if not isinstance(item, ToolCall):
                    raise ValueError(f"All items in tools list must be ToolCall instances, got {type(item)}")
            return Tools(tools=v)
        
        raise ValueError(f"tools must be either a Tools object or a list of ToolCall objects, got {type(v)}")

    @field_validator("backup_models")
    @classmethod
    def validate_backup_models(cls, v):
        """Validate the backup models list."""
        if not v:
            return v

        # Check for duplicate priorities
        priorities = [backup.priority for backup in v]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Backup models must have unique priorities")

        return v

    def get_ordered_backup_models(self) -> List[LLM]:
        """Return backup models ordered by priority (lowest priority number first)."""
        return [
            backup.model
            for backup in sorted(self.backup_models, key=lambda x: x.priority)
        ]
