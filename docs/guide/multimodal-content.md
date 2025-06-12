# Multimodal Content

v-router provides seamless support for multimodal content across all providers. Send images, PDFs, and Word documents with automatic format conversion and unified handling.

## Overview

v-router supports three types of multimodal content:

- **Images**: JPEG, PNG, GIF, WebP formats
- **PDFs**: Full document support (where provider supports it)
- **Word Documents**: .docx files with automatic HTML conversion

All content is automatically detected and converted based on file paths or MIME types, providing a consistent interface across all providers.

## Supported Content Types

### Images
All providers support images with automatic base64 encoding:

```python
from v_router import Client, LLM
from v_router.classes.message import ImageContent

# Method 1: File path (automatic detection)
messages = [
    {"role": "user", "content": "/path/to/image.jpg"}
]

# Method 2: Base64 encoded
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What's in this image?"},
            ImageContent(data=base64_image_data, media_type="image/jpeg")
        ]
    }
]
```

### PDF Documents
PDFs are supported by Anthropic and Google providers:

```python
from v_router.classes.message import DocumentContent

# Method 1: File path (automatic detection)
messages = [
    {"role": "user", "content": "/path/to/document.pdf"}
]

# Method 2: Base64 encoded
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this document"},
            DocumentContent(data=base64_pdf_data, media_type="application/pdf")
        ]
    }
]
```

### Word Documents (.docx)
Word documents are converted to HTML using mammoth and work across all providers:

```python
# Method 1: File path (automatic detection)
messages = [
    {"role": "user", "content": "/path/to/document.docx"}
]

# Method 2: Base64 encoded
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this Word document"},
            DocumentContent(
                data=base64_docx_data, 
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        ]
    }
]
```

## Provider Compatibility

| Provider | Images | PDFs | Word Docs | Notes |
|----------|--------|------|-----------|-------|
| **Anthropic** | ✅ | ✅ | ✅ | Full native support for PDFs |
| **OpenAI** | ✅ | ✅ | ✅ | PDF support via responses API |
| **Google AI** | ✅ | ✅ | ✅ | Full native support for PDFs |
| **Azure OpenAI** | ✅ | ✅ | ✅ | PDF support via responses API |
| **Vertex AI** | ✅ | ✅ | ✅ | Full support via Google infrastructure |

!!! note "Provider-Specific Handling"
    - **PDF Support**: OpenAI providers show a placeholder message for PDFs
    - **Word Documents**: All providers receive Word content as converted HTML text
    - **Images**: Universally supported across all providers

## Automatic File Detection

v-router automatically detects file types based on extensions:

```python
# These are all detected automatically
messages = [
    {"role": "user", "content": "/path/to/photo.jpg"},      # → ImageContent
    {"role": "user", "content": "/path/to/document.pdf"},   # → DocumentContent
    {"role": "user", "content": "/path/to/report.docx"},    # → TextContent (converted)
]
```

Supported extensions:
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.tiff`, `.tif`, `.bmp`
- **PDFs**: `.pdf`
- **Word Documents**: `.docx`

## Complex Multimodal Messages

Combine multiple content types in a single message:

```python
from v_router.classes.message import TextContent, ImageContent, DocumentContent

messages = [
    {
        "role": "user",
        "content": [
            TextContent(text="Please analyze these documents:"),
            TextContent(text="1. Project overview (Word):"),
            DocumentContent(data=docx_data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            TextContent(text="2. Supporting image:"),
            ImageContent(data=image_data, media_type="image/jpeg"),
            TextContent(text="3. Technical specifications (PDF):"),
            DocumentContent(data=pdf_data, media_type="application/pdf"),
            TextContent(text="What are the key insights from these materials?")
        ]
    }
]
```

## Cross-Provider Example

The same multimodal content works across all providers:

```python
import asyncio
from v_router import Client, LLM
from v_router.classes.message import TextContent, ImageContent

async def test_multimodal():
    # Prepare multimodal content
    messages = [
        {
            "role": "user",
            "content": [
                TextContent(text="Describe what you see in this image:"),
                ImageContent(data=image_data, media_type="image/jpeg")
            ]
        }
    ]
    
    # Test across multiple providers
    providers = [
        ("anthropic", "claude-sonnet-4"),
        ("openai", "gpt-4o"),
        ("google", "gemini-1.5-pro")
    ]
    
    for provider, model in providers:
        llm_config = LLM(model_name=model, provider=provider)
        client = Client(llm_config)
        
        try:
            response = await client.messages.create(messages=messages)
            print(f"{provider}: {response.content[0].text[:100]}...")
        except Exception as e:
            print(f"{provider}: Error - {e}")

asyncio.run(test_multimodal())
```

## Best Practices

### File Size Optimization
- **Images**: Resize to reasonable dimensions (e.g., 1024x1024) before sending
- **PDFs**: Keep file size under 5MB for best performance
- **Word Documents**: Complex formatting may not convert perfectly to HTML

### Error Handling
```python
try:
    response = await client.messages.create(messages=messages)
except Exception as e:
    print(f"Multimodal request failed: {e}")
    # v-router will have attempted fallback providers if configured
```

### Provider Selection
```python
# Use providers that support your content type
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",  # Supports PDFs natively
    backup_models=[
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=1  # Also supports PDFs
        )
    ]
)
```

### Content Validation
```python
from pathlib import Path
import mimetypes

def validate_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    mime_type, _ = mimetypes.guess_type(str(path))
    supported_types = [
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    if mime_type not in supported_types:
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    return mime_type

# Use before sending
try:
    mime_type = validate_file("document.docx")
    messages = [{"role": "user", "content": "document.docx"}]
except (FileNotFoundError, ValueError) as e:
    print(f"File validation failed: {e}")
```

## Advanced Usage

### Manual Content Creation
For fine-grained control, create content objects manually:

```python
import base64
from v_router.classes.message import TextContent, ImageContent, DocumentContent

# Read and encode files manually
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

with open("document.docx", "rb") as f:
    docx_data = base64.b64encode(f.read()).decode("utf-8")

# Create content objects
content = [
    TextContent(text="Compare these documents:"),
    ImageContent(data=image_data, media_type="image/jpeg"),
    DocumentContent(
        data=docx_data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
]

messages = [{"role": "user", "content": content}]
```

### Fallback Strategies
Configure provider fallbacks based on content type support:

```python
# Primary: Anthropic (supports all types)
# Backup: Google (supports all types)
# Final: OpenAI (images and Word docs only)
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=1
        ),
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=2
        )
    ]
)
```

## Next Steps

- **[Function Calling](function-calling.md)**: Combine multimodal input with tool usage
- **[Configuration](../getting-started/configuration.md)**: Set up provider-specific options
- **[Examples](../examples/basic.md)**: See complete working examples

[Explore Function Calling →](function-calling.md){ .md-button .md-button--primary }
[View Examples →](../examples/basic.md){ .md-button }