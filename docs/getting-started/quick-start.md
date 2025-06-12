# Quick Start

Get up and running with v-router in minutes. This guide covers the basic usage patterns and core concepts.

## Your First Request

Let's start with a simple example that shows v-router's core functionality:

```python
import asyncio
from v_router import Client, LLM

async def main():
    # Create an LLM configuration
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        max_tokens=100,
        temperature=0.7
    )
    
    # Create a client
    client = Client(llm_config)
    
    # Send a message
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": "Hello! Explain quantum computing in one sentence."}
        ]
    )
    
    print(f"Response: {response.content[0].text}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")

# Run the example
asyncio.run(main())
```

!!! note "API Keys Required"
    Make sure you have your API keys configured before running this example. See the [Configuration](configuration.md) guide for details.

## Key Concepts

### LLM Configuration

The `LLM` class defines your model configuration:

```python
from v_router import LLM

llm_config = LLM(
    model_name="claude-sonnet-4",  # Model to use
    provider="anthropic",          # Provider to use
    max_tokens=1000,              # Maximum tokens to generate
    temperature=0.7,              # Creativity level (0-1)
    system_prompt="You are a helpful assistant"  # System message
)
```

### Client and Messages

The `Client` provides the main interface, with a `messages` API similar to provider SDKs:

```python
from v_router import Client

client = Client(llm_config)

# The messages API matches OpenAI/Anthropic patterns
response = await client.messages.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What's the weather like?"}
    ]
)
```

### Response Format

All providers return the same unified response format:

```python
# Response object
print(response.content[0].text)    # The generated text
print(response.model)              # Model that was actually used
print(response.provider)           # Provider that was used
print(response.usage.total_tokens) # Token usage information

# Access the raw provider response if needed
print(response.raw_response)
```

## Adding Automatic Fallback

One of v-router's key features is automatic fallback when your primary model fails:

```python
from v_router import Client, LLM, BackupModel

# Configure primary model with backups
llm_config = LLM(
    model_name="claude-6",  # This might fail (doesn't exist yet)
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1  # Try this first as backup
        ),
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2  # Try this second
        )
    ]
)

client = Client(llm_config)

# If claude-6 fails, automatically tries gpt-4o, then gemini-1.5-pro
response = await client.messages.create(
    messages=[{"role": "user", "content": "What's 2+2?"}]
)

print(f"Successfully used: {response.model} from {response.provider}")
```

## Cross-Provider Switching

Enable automatic switching between providers for the same model:

```python
llm_config = LLM(
    model_name="claude-opus-4",
    provider="vertexai",  # Try Vertex AI first
    try_other_providers=True  # Fall back to direct Anthropic if needed
)

client = Client(llm_config)

# Will try Vertex AI first, then Anthropic directly
response = await client.messages.create(
    messages=[{"role": "user", "content": "Tell me a joke."}]
)
```

## Working with Different Providers

v-router automatically maps model names across providers. Here are some examples:

=== "Anthropic"

    ```python
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic"
    )
    ```

=== "OpenAI"

    ```python
    llm_config = LLM(
        model_name="gpt-4o",
        provider="openai"
    )
    ```

=== "Google AI"

    ```python
    llm_config = LLM(
        model_name="gemini-1.5-pro",
        provider="google"
    )
    ```

=== "Vertex AI"

    ```python
    llm_config = LLM(
        model_name="claude-sonnet-4",  # Same model name
        provider="vertexai"            # Different provider
    )
    ```

## Error Handling

v-router provides detailed error information when requests fail:

```python
try:
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
except Exception as e:
    print(f"Request failed: {e}")
    # v-router will have already tried backup models if configured
```

## Function Calling Preview

v-router provides unified function calling across all providers with fine-grained control:

```python
from v_router.classes.tools import ToolCall, Tools
from pydantic import BaseModel, Field

# Define a tool
class Calculator(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

calc_tool = ToolCall(
    name="calculator",
    description="Perform calculations",
    input_schema=Calculator.model_json_schema()
)

# Configure with tool control
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=Tools(tools=[calc_tool]),
    tool_choice="auto"  # "auto", "any", "none", or tool name
)

client = Client(llm_config)
response = await client.messages.create(
    messages=[{"role": "user", "content": "What's 15 * 23?"}]
)

# Check for tool calls
if response.tool_use:
    for tool_call in response.tool_use:
        print(f"Tool: {tool_call.name}, Args: {tool_call.arguments}")
```

!!! tip "Tool Choice Control"
    Use `tool_choice="calculator"` to force the calculator tool, `"any"` to require any tool, or `"none"` to disable tools entirely.

## Multimodal Content Preview

v-router supports images, PDFs, and Word documents across all providers with automatic format conversion:

```python
# Send an image by file path (automatic detection)
response = await client.messages.create(
    messages=[
        {"role": "user", "content": "/path/to/image.jpg"},
        {"role": "user", "content": "What do you see in this image?"}
    ]
)

# Send a Word document (automatically converted to HTML)
response = await client.messages.create(
    messages=[
        {"role": "user", "content": "/path/to/document.docx"},
        {"role": "user", "content": "Summarize this document"}
    ]
)

# Combine multiple content types
from v_router.classes.message import TextContent, ImageContent, DocumentContent

messages = [
    {
        "role": "user",
        "content": [
            TextContent(text="Analyze these materials:"),
            ImageContent(data=base64_image, media_type="image/jpeg"),
            DocumentContent(data=base64_docx, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        ]
    }
]
```

!!! note "Cross-Provider Support"
    - **Images**: Supported by all providers
    - **PDFs**: Supported by Anthropic, Google, and Vertex AI
    - **Word Documents**: Converted to HTML and supported by all providers

## Complete Example

Here's a complete example that demonstrates multiple features:

```python
import asyncio
import os
from v_router import Client, LLM, BackupModel

async def demo():
    # Configure with multiple fallback options
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        max_tokens=500,
        temperature=0.8,
        system_prompt="You are a creative writing assistant",
        backup_models=[
            BackupModel(
                model=LLM(
                    model_name="gpt-4o", 
                    provider="openai",
                    max_tokens=500,
                    temperature=0.8
                ),
                priority=1
            )
        ],
        try_other_providers=True
    )
    
    client = Client(llm_config)
    
    # Send a creative writing request
    response = await client.messages.create(
        messages=[
            {
                "role": "user", 
                "content": "Write a short story about a robot learning to paint"
            }
        ]
    )
    
    print("=== Creative Story ===")
    print(response.content[0].text)
    print(f"\nGenerated by: {response.model} ({response.provider})")
    print(f"Tokens used: {response.usage.total_tokens}")

# Make sure you have API keys set
if __name__ == "__main__":
    # Check for required environment variables
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Please set these environment variables: {missing_keys}")
        print("See the configuration guide for details.")
    else:
        asyncio.run(demo())
```

## Next Steps

Now that you understand the basics, explore more advanced features:

- **[Function Calling](../guide/function-calling.md)**: Use tools and function calls
- **[Multimodal Content](../guide/multimodal-content.md)**: Send images, PDFs, and Word documents
- **[Configuration](configuration.md)**: Set up API keys and advanced options
- **[Examples](../examples/basic.md)**: See more detailed examples

[Explore Function Calling →](../guide/function-calling.md){ .md-button .md-button--primary }
[See More Examples →](../examples/basic.md){ .md-button }