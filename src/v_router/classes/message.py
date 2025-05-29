from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message object in a chat conversation."""

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Role of the message sender, can be 'user', 'assistant', or 'system'.",
    )
    content: str = Field(
        ...,
        description="Content of the message, which can be text or other data types.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        """Validate and convert content to string if necessary."""
        if isinstance(v, str):
            return v

        try:
            # Attempt to convert to string
            converted = str(v)

            # Check if conversion resulted in something meaningful
            if converted in ["None", "", "nan"]:
                raise ValueError(
                    f"Content conversion resulted in invalid value: {converted}"
                )

            return converted
        except Exception as e:
            raise ValueError(f"Unable to convert content to string: {e}")
