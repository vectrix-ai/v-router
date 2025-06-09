# LLM Class Reference

The `LLM` class is the core configuration object in v-router that defines how your Large Language Model requests should be handled. It provides a unified interface for configuring models across different providers with support for fallback strategies and advanced parameters.

## Class Definition

```python
from v_router import LLM, BackupModel

llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    max_tokens=1000,
    temperature=0.7,
    backup_models=[...],
    try_other_providers=True,
    tools=None,
    tool_choice=None
)
```

## Parameters

### Required Parameters

#### `model_name: str`
**Description**: Name of the LLM model to be used.

**Examples**:
```python
# Anthropic models
LLM(model_name="claude-sonnet-4", provider="anthropic")
LLM(model_name="claude-opus-4", provider="anthropic")
LLM(model_name="claude-haiku-4", provider="anthropic")

# OpenAI models
LLM(model_name="gpt-4o", provider="openai")
LLM(model_name="gpt-4", provider="openai")
LLM(model_name="gpt-3.5", provider="openai")

# Google models
LLM(model_name="gemini-1.5-pro", provider="google")
LLM(model_name="gemini-1.5-flash", provider="google")
LLM(model_name="gemini-2.0-flash", provider="google")
```

#### `provider: Literal["openai", "anthropic", "azure", "google", "vertexai"]`
**Description**: Provider of the LLM service.

**Supported Values**:
- `"openai"` - OpenAI API
- `"anthropic"` - Anthropic API (direct)
- `"azure"` - Azure OpenAI Service
- `"google"` - Google AI Studio
- `"vertexai"` - Google Cloud Vertex AI

**Examples**:
```python
# Direct provider APIs
LLM(model_name="claude-sonnet-4", provider="anthropic")
LLM(model_name="gpt-4o", provider="openai")
LLM(model_name="gemini-1.5-pro", provider="google")

# Cloud platforms
LLM(model_name="gpt-4", provider="azure")
LLM(model_name="claude-sonnet-4", provider="vertexai")
```

### Optional Parameters

#### `max_tokens: Optional[int] = None`
**Description**: Maximum number of tokens to generate in the response. If not specified, defaults to the model's maximum.

**Examples**:
```python
# Short responses
LLM(model_name="claude-sonnet-4", provider="anthropic", max_tokens=100)

# Long responses
LLM(model_name="gpt-4o", provider="openai", max_tokens=4000)

# Use model default
LLM(model_name="gemini-1.5-pro", provider="google")  # max_tokens=None
```

#### `temperature: Optional[float] = 0`
**Description**: Sampling temperature for the model. Higher values (up to 1.0) make output more random, lower values make it more deterministic.

**Range**: 0.0 to 1.0

**Examples**:
```python
# Deterministic output
LLM(model_name="claude-sonnet-4", provider="anthropic", temperature=0.0)

# Balanced creativity
LLM(model_name="gpt-4o", provider="openai", temperature=0.7)

# Highly creative
LLM(model_name="gemini-1.5-pro", provider="google", temperature=0.9)

# Creative writing
LLM(
    model_name="claude-opus-4",
    provider="anthropic",
    temperature=0.8,
    max_tokens=2000
)
```

#### `backup_models: List[BackupModel] = []`
**Description**: List of backup models with priorities. Models will be tried in order of priority (lowest number first) if the primary model fails.

**Examples**:
```python
from v_router import LLM, BackupModel

# Single backup
llm_config = LLM(
    model_name="claude-6",  # Primary (might not exist)
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        )
    ]
)

# Multiple backups with priorities
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1  # First fallback
        ),
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2  # Second fallback
        ),
        BackupModel(
            model=LLM(
                model_name="gpt-3.5",
                provider="openai",
                max_tokens=1000
            ),
            priority=3  # Final fallback
        )
    ]
)

# Cross-provider same model backup
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="vertexai",  # Primary: Vertex AI
    backup_models=[
        BackupModel(
            model=LLM(model_name="claude-sonnet-4", provider="anthropic"),
            priority=1  # Fallback: Direct Anthropic
        )
    ]
)
```

#### `try_other_providers: bool = False`
**Description**: Try other providers for this model as backup if available. Only the exact same model will be used for this cross-provider fallback.

**Examples**:
```python
# Enable cross-provider fallback
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="vertexai",  # Try Vertex AI first
    try_other_providers=True  # Fall back to Anthropic if needed
)

# Combine with backup models
llm_config = LLM(
    model_name="claude-opus-4",
    provider="anthropic",
    try_other_providers=True,  # Try other providers for claude-opus-4
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1  # Different model as final backup
        )
    ]
)
```

#### `tools: Optional[Tools] = None`
**Description**: Tools/functions that can be called by the model. Can be a `Tools` object or automatically converted from a list of `ToolCall` objects.

**Examples**:
```python
from pydantic import BaseModel, Field
from v_router import LLM, ToolCall, Tools

# Define tool schema
class WeatherQuery(BaseModel):
    location: str = Field(..., description="City and state, e.g. San Francisco, CA")
    units: str = Field("fahrenheit", description="Temperature units")

# Create tool
weather_tool = ToolCall(
    name="get_weather",
    description="Get current weather for a location",
    input_schema=WeatherQuery.model_json_schema()
)

# Method 1: Using Tools object
tools_obj = Tools(tools=[weather_tool])
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=tools_obj
)

# Method 2: Using list (auto-converted to Tools)
llm_config = LLM(
    model_name="gpt-4o",
    provider="openai",
    tools=[weather_tool]  # Automatically wrapped in Tools object
)

# Multiple tools
search_tool = ToolCall(
    name="web_search",
    description="Search the web for information",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
)

llm_config = LLM(
    model_name="gemini-1.5-pro",
    provider="google",
    tools=[weather_tool, search_tool]
)
```

#### `tool_choice: Optional[Union[str, Dict[str, Any]]] = None`
**Description**: Controls how the model uses tools.

**Options**:
- `None` or `"auto"`: Model decides whether to use tools (default)
- `"any"`: Model must use one of the provided tools
- `"none"`: Model is prevented from using tools
- `str` (tool name): Force the model to use a specific tool
- `dict`: Provider-specific format for advanced control

**Examples**:
```python
# Let model decide (default)
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=[weather_tool],
    tool_choice=None  # or "auto"
)

# Force model to use any tool
llm_config = LLM(
    model_name="gpt-4o",
    provider="openai",
    tools=[weather_tool, search_tool],
    tool_choice="any"  # Must use one of the tools
)

# Force specific tool
llm_config = LLM(
    model_name="gemini-1.5-pro",
    provider="google",
    tools=[weather_tool, search_tool],
    tool_choice="get_weather"  # Must use weather tool
)

# Prevent tool use
llm_config = LLM(
    model_name="claude-haiku-4",
    provider="anthropic",
    tools=[weather_tool],
    tool_choice="none"  # Don't use tools
)

# Provider-specific control (Advanced)
llm_config = LLM(
    model_name="gpt-4o",
    provider="openai",
    tools=[weather_tool],
    tool_choice={
        "type": "function",
        "function": {"name": "get_weather"}
    }
)
```

## BackupModel Class

The `BackupModel` class is used to configure fallback models with priorities.

### Parameters

#### `model: LLM`
**Description**: The backup LLM model configuration.

#### `priority: int`
**Description**: Priority of this backup model. Lower numbers = higher priority (1 is highest priority).

**Validation**: Must be at least 1, and all backup models must have unique priorities.

**Examples**:
```python
from v_router import LLM, BackupModel

# Single priority backup
backup = BackupModel(
    model=LLM(model_name="gpt-4o", provider="openai"),
    priority=1
)

# Multiple backups with different priorities
backups = [
    BackupModel(
        model=LLM(model_name="gpt-4o", provider="openai"),
        priority=1  # Try first
    ),
    BackupModel(
        model=LLM(model_name="gemini-1.5-pro", provider="google"),
        priority=2  # Try second
    ),
    BackupModel(
        model=LLM(model_name="gpt-3.5", provider="openai"),
        priority=3  # Try last
    )
]

llm_config = LLM(
    model_name="claude-ultra-5",  # Primary (doesn't exist)
    provider="anthropic",
    backup_models=backups
)
```

## Complete Configuration Examples

### Basic Configuration
```python
from v_router import LLM, Client

# Simple configuration
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic"
)

client = Client(llm_config)
```

### Production Configuration with Fallbacks
```python
from v_router import LLM, BackupModel, Client

# Robust production configuration
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    max_tokens=2000,
    temperature=0.3,
    try_other_providers=True,
    backup_models=[
        BackupModel(
            model=LLM(
                model_name="gpt-4o",
                provider="openai",
                max_tokens=2000,
                temperature=0.3
            ),
            priority=1
        ),
        BackupModel(
            model=LLM(
                model_name="gemini-1.5-pro",
                provider="google",
                max_tokens=2000,
                temperature=0.3
            ),
            priority=2
        )
    ]
)

client = Client(llm_config)
```

### Function Calling Configuration
```python
from pydantic import BaseModel, Field
from v_router import LLM, ToolCall, Client

# Define tool schema
class CalculatorInput(BaseModel):
    operation: str = Field(..., description="Math operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

# Create tool
calculator_tool = ToolCall(
    name="calculator",
    description="Perform basic math operations",
    input_schema=CalculatorInput.model_json_schema()
)

# Configure with tools
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=[calculator_tool],
    tool_choice="auto",
    backup_models=[
        BackupModel(
            model=LLM(
                model_name="gpt-4o",
                provider="openai",
                # Tools are automatically inherited by backup models
            ),
            priority=1
        )
    ]
)

client = Client(llm_config)
```

### Creative Writing Configuration
```python
from v_router import LLM, BackupModel, Client

# Configuration optimized for creative tasks
llm_config = LLM(
    model_name="claude-opus-4",
    provider="anthropic",
    max_tokens=4000,
    temperature=0.8,  # High creativity
    backup_models=[
        BackupModel(
            model=LLM(
                model_name="gpt-4o",
                provider="openai",
                max_tokens=4000,
                temperature=0.8
            ),
            priority=1
        )
    ]
)

client = Client(llm_config)
```

## Methods

### `get_ordered_backup_models() -> List[LLM]`
**Description**: Return backup models ordered by priority (lowest priority number first).

**Returns**: List of `LLM` objects sorted by priority.

**Example**:
```python
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2
        ),
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        )
    ]
)

# Get ordered backup models
ordered_backups = llm_config.get_ordered_backup_models()
# Returns: [LLM(model_name="gpt-4o", ...), LLM(model_name="gemini-1.5-pro", ...)]
```

## Usage with Client

Once configured, use the `LLM` object with the `Client` to make requests:

```python
from v_router import Client

# Create client with LLM configuration
client = Client(llm_config)

# Make requests
response = await client.messages.create(
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(f"Response: {response.content[0].text}")
print(f"Model used: {response.model}")
print(f"Provider used: {response.provider}")
```

## Best Practices

### 1. Use Appropriate Models for Tasks
```python
# Fast responses for simple tasks
quick_llm = LLM(model_name="gemini-1.5-flash", provider="google")

# High-quality responses for complex tasks
quality_llm = LLM(model_name="claude-opus-4", provider="anthropic")

# Balanced performance
balanced_llm = LLM(model_name="claude-sonnet-4", provider="anthropic")
```

### 2. Configure Appropriate Fallbacks
```python
# Same model, different providers
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="vertexai",
    try_other_providers=True  # Falls back to Anthropic
)

# Different models as fallbacks
llm_config = LLM(
    model_name="claude-opus-4",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="claude-sonnet-4", provider="anthropic"),
            priority=1
        ),
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=2
        )
    ]
)
```

### 3. Set Consistent Parameters Across Backups
```python
# Maintain consistency across primary and backup models
base_params = {
    "max_tokens": 1000,
    "temperature": 0.7
}

llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    **base_params,
    backup_models=[
        BackupModel(
            model=LLM(
                model_name="gpt-4o",
                provider="openai",
                **base_params  # Same parameters
            ),
            priority=1
        )
    ]
)
```

### 4. Use Environment-Specific Configurations
```python
import os

def get_llm_config():
    if os.getenv("ENVIRONMENT") == "production":
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=2000,
            temperature=0.3,
            try_other_providers=True,
            backup_models=[
                BackupModel(
                    model=LLM(model_name="gpt-4o", provider="openai"),
                    priority=1
                )
            ]
        )
    else:
        return LLM(
            model_name="gemini-1.5-flash",
            provider="google",
            max_tokens=500
        )
```

## See Also

- [Client API Reference](client.md) - Learn how to use the LLM configuration with the Client
- [Function Calling Guide](../guide/function-calling.md) - Using tools with LLM configurations
- [Configuration Guide](../getting-started/configuration.md) - Environment setup and API keys
- [Basic Examples](../examples/basic.md) - Simple usage examples