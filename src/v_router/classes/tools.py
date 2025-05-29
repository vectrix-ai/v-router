from typing import List

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A function call to a tool."""

    name: str = Field(..., description="The name of the function to call.")
    description: str = Field(
        ..., description="The description of the function to call."
    )
    input_schema: BaseModel = Field(
        ..., description="The schema of the input to the function."
    )


class Tools(BaseModel):
    """A list of tools."""

    tools: List[ToolCall] = Field(..., description="The list of tools.")
