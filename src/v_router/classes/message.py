from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message object in a chat conversation."""
    
    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Role of the message sender, can be 'user', 'assistant', or 'system'."
    )
    content: str = Field(
        ...,
        description="Content of the message, which can be text or other data types."
    )

