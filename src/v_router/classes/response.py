from typing import Literal

from pydantic import BaseModel, Field


class Content(BaseModel):
    """The content of the response."""

    type: Literal["text", "tool_use"] = Field(
        ...,
        description="The type of the content.",
    )
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="The role of the content."
    )
    text: str = Field(..., description="The text of the content.")


class ToolCall(BaseModel):
    """A tool use object from the LLM."""

    id: str = Field(..., description="The ID of the tool use.")
    args: dict = Field(..., description="The arguments of the tool use.")
    name: str = Field(..., description="The name of the tool used.")


class Usage(BaseModel):
    """A usage object from the LLM."""

    input_tokens: int | None = Field(
        None, description="The number of input tokens used."
    )
    output_tokens: int | None = Field(
        None, description="The number of output tokens used."
    )


class AIMessage(BaseModel):
    """An AI message response from the LLM."""

    id: str | None = Field(None, description="The ID of the message.")
    content: list[Content] = Field(..., description="The content of the response.")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="The tool use of the response."
    )
    usage: Usage = Field(..., description="The usage of the response.")
    model: str = Field(..., description="The model used to generate the response.")
    provider: str = Field(
        ..., description="The provider used to generate the response."
    )
    raw_response: dict = Field(..., description="The raw response from the LLM.")
