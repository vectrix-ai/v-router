import base64
import mimetypes
from pathlib import Path
from typing import List, Literal, Union

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

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Role of the message sender, can be 'user', 'assistant', or 'system'.",
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
