import base64
import mimetypes
from pathlib import Path
from typing import Any, List, Literal, Union

import mammoth
from pydantic import BaseModel, Field, field_validator, model_validator


# Content types for multimodal messages
class TextContent(BaseModel):
    """Text content in a message."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content in a message."""

    type: Literal["image"] = "image"
    data: str  # Base64 encoded image data
    media_type: str  # MIME type (e.g., "image/jpeg")


class DocumentContent(BaseModel):
    """Document content in a message (PDF or Word)."""

    type: Literal["document"] = "document"
    data: str  # Base64 encoded document data
    media_type: str = "application/pdf"

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, v: str) -> str:
        """Validate that the media type is supported."""
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]
        if v not in allowed_types:
            raise ValueError(
                f"Unsupported media type: {v}. Allowed types: {allowed_types}"
            )
        return v


ContentType = Union[TextContent, ImageContent, DocumentContent]


class Message(BaseModel):
    """A single message object in a chat conversation."""

    role: Literal["user", "assistant", "system", "tool"] = Field(
        ...,
        description="Role of the message sender, can be 'user', 'assistant', 'system', or 'tool'.",
    )
    content: Union[str, List[ContentType]] = Field(
        ...,
        description="Content of the message, which can be text, images, or documents.",
    )

    @model_validator(mode="before")
    @classmethod
    def process_content(cls, values: dict) -> dict:
        """Process content and handle file paths for images and PDFs."""
        content = values.get("content")

        if content is None:
            return values

        # If content is already a list or a proper content type, return as is
        if isinstance(content, list):
            return values

        # Convert numeric content to string
        if isinstance(content, int | float):
            values["content"] = str(content)
            return values

        # If content is a string, check if it's a file path
        if isinstance(content, str):
            # Only check for file path if the string is not too long and looks like a path
            # Check for common file extensions or absolute/relative paths
            if len(content) < 500 and (
                "/" in content
                or "\\" in content
                or content.startswith(".")
                or any(
                    content.lower().endswith(ext)
                    for ext in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".webp",
                        ".pdf",
                        ".docx",
                        ".tiff",
                        ".tif",
                        ".bmp",
                    ]
                )
            ):
                try:
                    path = Path(content)
                    if path.exists() and path.is_file():
                        # Determine file type
                        mime_type, _ = mimetypes.guess_type(str(path))

                        if mime_type:
                            # Read file and encode to base64
                            with open(path, "rb") as f:
                                file_data = base64.b64encode(f.read()).decode("utf-8")

                            if mime_type.startswith("image/"):
                                # Check allowed image types
                                allowed_types = [
                                    "image/jpeg",
                                    "image/jpg",
                                    "image/png",
                                    "image/gif",
                                    "image/webp",
                                ]
                                if mime_type not in allowed_types:
                                    raise ValueError(
                                        f"Unsupported image type: {mime_type}. Allowed types: {allowed_types}"
                                    )

                                values["content"] = [
                                    ImageContent(data=file_data, media_type=mime_type)
                                ]
                            elif mime_type == "application/pdf":
                                values["content"] = [DocumentContent(data=file_data)]
                            elif (
                                mime_type
                                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            ):
                                # Convert Word document to HTML using mammoth
                                try:
                                    with open(path, "rb") as docx_file:
                                        result = mammoth.convert_to_html(docx_file)
                                        html_content = result.value
                                    # Return as text content since Word docs are converted to HTML
                                    values["content"] = [TextContent(text=html_content)]
                                except Exception:
                                    # If conversion fails, store as document for manual handling
                                    values["content"] = [
                                        DocumentContent(
                                            data=file_data,
                                            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        )
                                    ]
                            else:
                                # Not an image or PDF, treat as text
                                values["content"] = content
                        else:
                            # Could not determine mime type, treat as text
                            values["content"] = content
                except OSError:
                    # If path operations fail, treat as regular text
                    pass
                except ValueError:
                    # Re-raise ValueError for validation errors (like unsupported image types)
                    raise
            # else: regular text content, keep as is

        return values

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_format(
        cls, v: Union[str, List[ContentType]]
    ) -> Union[str, List[ContentType]]:
        """Ensure content is in the correct format."""
        # Keep string content as string for backward compatibility
        # Only convert to list if it's already a list
        return v

    def get_text_content(self) -> str:
        """Extract text content from the message."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = []
            for item in self.content:
                if isinstance(item, TextContent):
                    text_parts.append(item.text)
                elif hasattr(item, "type"):
                    text_parts.append(f"[{item.type}]")
            return " ".join(text_parts)
        return ""


class HumanMessage(Message):
    """Message from a human user.

    HumanMessages are messages that are passed in from a human to the model.
    The role is always set to "user" and cannot be changed.

    Args:
        content: The content of the message as a positional argument.
        **kwargs: Additional fields to pass to the message.

    Returns:
        A HumanMessage object.

    """

    role: Literal["user"] = Field(
        default="user",
        description="Role is always 'user' for HumanMessage.",
    )
    name: Union[str, None] = Field(
        default=None,
        description="An optional name for the message. This can be used to provide a human-readable name.",
    )
    id: Union[str, None] = Field(
        default=None,
        description="An optional unique identifier for the message.",
    )
    example: bool = Field(
        default=False,
        description="Use to denote that a message is part of an example conversation. Usage is discouraged.",
    )

    def __init__(self, content: Union[str, List[ContentType], None] = None, **kwargs):
        """Initialize a HumanMessage.

        Args:
            content: The content of the message as a positional argument.
            **kwargs: Additional fields to pass to the message.

        """
        if content is not None:
            kwargs["content"] = content
        kwargs["role"] = "user"  # Always set role to "user"
        super().__init__(**kwargs)

    def text(self) -> str:
        """Get the text content of the message.

        Returns:
            The text content of the message.

        """
        return self.get_text_content()


class SystemMessage(Message):
    """Message from the system.

    SystemMessages are messages that provide context or instructions to the model.
    The role is always set to "system" and cannot be changed.

    Args:
        content: The content of the message as a positional argument.
        **kwargs: Additional fields to pass to the message.

    Returns:
        A SystemMessage object.

    """

    role: Literal["system"] = Field(
        default="system",
        description="Role is always 'system' for SystemMessage.",
    )
    name: Union[str, None] = Field(
        default=None,
        description="An optional name for the message. This can be used to provide a human-readable name.",
    )
    id: Union[str, None] = Field(
        default=None,
        description="An optional unique identifier for the message.",
    )

    def __init__(self, content: Union[str, List[ContentType], None] = None, **kwargs):
        """Initialize a SystemMessage.

        Args:
            content: The content of the message as a positional argument.
            **kwargs: Additional fields to pass to the message.

        """
        if content is not None:
            kwargs["content"] = content
        kwargs["role"] = "system"  # Always set role to "system"
        super().__init__(**kwargs)

    def text(self) -> str:
        """Get the text content of the message.

        Returns:
            The text content of the message.

        """
        return self.get_text_content()


class ToolMessage(Message):
    """Message from a tool.

    ToolMessages are messages that are passed in from a tool to the model.
    The role is always set to "tool" and cannot be changed.

    Args:
        content: The content of the message as a positional argument.
        tool_call_id: Tool call that this message is responding to.
        status: Status of the tool invocation. Defaults to 'success'.
        artifact: Optional artifact of the Tool execution which is not meant to be sent to the model.
        **kwargs: Additional fields to pass to the message.

    Returns:
        A ToolMessage object.

    """

    role: Literal["tool"] = Field(
        default="tool",
        description="Role is always 'tool' for ToolMessage.",
    )
    tool_call_id: str = Field(
        ...,
        description="Tool call that this message is responding to.",
    )
    status: Literal["success", "error"] = Field(
        default="success",
        description="Status of the tool invocation.",
    )
    name: Union[str, None] = Field(
        default=None,
        description="An optional name for the message. This can be used to provide a human-readable name.",
    )
    id: Union[str, None] = Field(
        default=None,
        description="An optional unique identifier for the message.",
    )
    artifact: Any = Field(
        default=None,
        description="Artifact of the Tool execution which is not meant to be sent to the model.",
    )

    def __init__(
        self,
        content: Union[str, List[ContentType], None] = None,
        tool_call_id: str = None,
        status: Literal["success", "error"] = "success",
        artifact: Any = None,
        **kwargs,
    ):
        """Initialize a ToolMessage.

        Args:
            content: The content of the message as a positional argument.
            tool_call_id: Tool call that this message is responding to.
            status: Status of the tool invocation. Defaults to 'success'.
            artifact: Optional artifact of the Tool execution.
            **kwargs: Additional fields to pass to the message.

        """
        if content is not None:
            kwargs["content"] = content
        if tool_call_id is not None:
            kwargs["tool_call_id"] = tool_call_id
        if status is not None:
            kwargs["status"] = status
        if artifact is not None:
            kwargs["artifact"] = artifact
        kwargs["role"] = "tool"  # Always set role to "tool"
        super().__init__(**kwargs)

    def text(self) -> str:
        """Get the text content of the message.

        Returns:
            The text content of the message.

        """
        return self.get_text_content()

    def pretty_repr(self, html: bool = False) -> str:
        """Get a pretty representation of the message.

        Args:
            html: Whether to format the message as HTML. If True, the message will be
                formatted with HTML tags. Default is False.

        Returns:
            A pretty representation of the message.

        """
        status_str = f" [{self.status}]" if self.status != "success" else ""
        tool_str = f"Tool (ID: {self.tool_call_id}){status_str}: "
        content_str = self.text()

        if html:
            return f"<div><strong>{tool_str}</strong>{content_str}</div>"
        return f"{tool_str}{content_str}"


# Response-related classes moved from response.py
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
    content: Union[str, List[str]] = Field(
        ..., description="The content of the response."
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="The tool use of the response."
    )
    usage: Usage = Field(..., description="The usage of the response.")
    model: str = Field(..., description="The model used to generate the response.")
    provider: str = Field(
        ..., description="The provider used to generate the response."
    )
    raw_response: dict = Field(..., description="The raw response from the LLM.")

    def get_text_content(self) -> str:
        """Extract text content from the message."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return " ".join(self.content)
        return ""
