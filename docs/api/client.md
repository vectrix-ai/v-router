# Client API

The `Client` class is the main entry point for v-router. It provides a unified interface for working with multiple LLM providers and handles automatic fallback when primary models fail.

## Client Class

::: v_router.Client
    handler: python
    options:
      show_source: false
      show_root_heading: true
      show_root_toc_entry: true

### Constructor

```python
from v_router import Client, LLM

llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    max_tokens=1000
)

client = Client(llm_config)
```

**Parameters:**

- `llm_config` (LLM): The LLM configuration object defining the model, provider, and parameters
- `**provider_kwargs`: Additional keyword arguments passed to the provider

## Messages API

The `messages` attribute provides access to the Messages API, which handles chat completions:

### messages.create()

Create a chat completion with automatic fallback support.

```python
response = await client.messages.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ],
    **kwargs
)
```

**Parameters:**

- `messages` (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys
- `**kwargs`: Additional parameters passed to the provider (e.g., temperature override)

**Returns:**

- [`Response`](response.md): Unified response object with content, usage, and metadata

## Usage Examples

### Basic Usage

```python
import asyncio
from v_router import Client, LLM

async def basic_example():
    llm_config = LLM(
        model_name="gpt-4o",
        provider="openai",
        max_tokens=500,
        temperature=0.7
    )
    
    client = Client(llm_config)
    
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": "Explain machine learning in simple terms"}
        ]
    )
    
    print(f"Response: {response.content[0].text}")
    print(f"Model used: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(basic_example())
```

### With Automatic Fallback

```python
from v_router import Client, LLM, BackupModel

async def fallback_example():
    llm_config = LLM(
        model_name="claude-6",  # This model doesn't exist yet
        provider="anthropic",
        backup_models=[
            BackupModel(
                model=LLM(model_name="gpt-4o", provider="openai"),
                priority=1
            ),
            BackupModel(
                model=LLM(model_name="gemini-1.5-pro", provider="google"),
                priority=2
            )
        ]
    )
    
    client = Client(llm_config)
    
    # Will automatically fallback to GPT-4o when Claude-6 fails
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(f"Successfully used: {response.model} from {response.provider}")

asyncio.run(fallback_example())
```

### Function Calling

```python
from v_router import Client, LLM
from v_router.classes.tools import ToolCall, Tools
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    location: str = Field(..., description="City and state")
    units: str = Field("fahrenheit", description="Temperature units")

async def function_calling_example():
    weather_tool = ToolCall(
        name="get_weather",
        description="Get weather information",
        input_schema=WeatherQuery.model_json_schema()
    )
    
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        tools=Tools(tools=[weather_tool])
    )
    
    client = Client(llm_config)
    
    response = await client.messages.create(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}]
    )
    
    if response.tool_use:
        for tool_call in response.tool_use:
            print(f"Tool: {tool_call.name}")
            print(f"Arguments: {tool_call.arguments}")

asyncio.run(function_calling_example())
```

### Multimodal Content

```python
async def multimodal_example():
    client = Client(
        LLM(model_name="claude-sonnet-4", provider="anthropic")
    )
    
    # Send image by file path (automatically converted)
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": "/path/to/image.jpg"},
            {"role": "user", "content": "Describe this image"}
        ]
    )
    
    print(response.content[0].text)

asyncio.run(multimodal_example())
```

### Provider-Specific Configuration

```python
async def provider_config_example():
    # OpenAI with custom parameters
    openai_config = LLM(
        model_name="gpt-4o",
        provider="openai",
        max_tokens=1000,
        temperature=0.8,
        openai_kwargs={
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
        }
    )
    
    # Anthropic with custom parameters
    anthropic_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        anthropic_kwargs={
            "stream": False,
            "extra_headers": {"Custom-Header": "value"}
        }
    )
    
    # Use either configuration
    client = Client(openai_config)
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )

asyncio.run(provider_config_example())
```

## Error Handling

The Client automatically handles many error scenarios through fallback mechanisms, but you can also implement your own error handling:

```python
async def error_handling_example():
    client = Client(
        LLM(model_name="claude-sonnet-4", provider="anthropic")
    )
    
    try:
        response = await client.messages.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Success: {response.content[0].text}")
        
    except Exception as e:
        print(f"All providers failed: {e}")
        # Handle the error appropriately

asyncio.run(error_handling_example())
```

## Best Practices

### Configuration Management

```python
# Use environment-specific configurations
def get_production_config():
    return LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        max_tokens=2000,
        temperature=0.3,  # Lower temperature for production
        backup_models=[
            BackupModel(
                model=LLM(model_name="gpt-4o", provider="openai"),
                priority=1
            )
        ]
    )

def get_development_config():
    return LLM(
        model_name="gpt-3.5",  # Cheaper for development
        provider="openai",
        max_tokens=500,
        temperature=0.7
    )
```

### Resource Management

```python
# Use context managers for resource cleanup
async def resource_management_example():
    async with Client(llm_config) as client:
        response = await client.messages.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        return response
```

### Monitoring and Logging

```python
import logging
from v_router import setup_logger

# Enable detailed logging
setup_logger("v_router", level=logging.INFO)

async def monitored_request():
    client = Client(llm_config)
    
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Log usage for monitoring
    logging.info(f"Request completed: {response.model}, tokens: {response.usage.total_tokens}")
    
    return response
```

## Related Documentation

- [LLM Configuration](llm.md): Configure models and providers
- [Messages API](messages.md): Message format and options
- [Response Format](response.md): Understanding response objects
- [Function Calling Guide](../guide/function-calling.md): Using tools with the client
- [Provider Configuration](../guide/provider-configuration.md): Provider-specific settings