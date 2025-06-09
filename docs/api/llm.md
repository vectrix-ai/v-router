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

## Provider-Specific Parameters

v-router uses a two-tier parameter system to support provider-specific features:

- **Core parameters**: Defined in the `LLM` class (model_name, provider, max_tokens, temperature, etc.)
- **Provider-specific parameters**: Passed via `**kwargs` when creating messages

This design allows you to use provider-specific features without modifying the core LLM class.

### How to Use Provider-Specific Parameters

Provider-specific parameters are passed when creating messages, NOT in the LLM configuration:

```python
from v_router import Client, LLM

# 1. Configure core parameters in LLM
client = Client(
    llm_config=LLM(
        model_name="claude-opus-4-20250514",
        provider="anthropic",
        max_tokens=32000,
        temperature=1
    )
)

# 2. Pass provider-specific parameters when creating messages
response = await client.messages.create(
    messages=[{"role": "user", "content": "Solve this complex problem"}],
    # Provider-specific parameters as kwargs:
    timeout=600,  # Anthropic timeout parameter
    thinking={    # Anthropic thinking parameter
        "type": "enabled",
        "budget_tokens": 10000
    }
)
```

### Common Provider-Specific Parameters

#### Anthropic
```python
# Extended timeout for long responses
response = await client.messages.create(
    messages=[...],
    timeout=600  # 10 minutes timeout
)

# Thinking mode (Claude Opus 4)
response = await client.messages.create(
    messages=[...],
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    }
)

# Top-k sampling
response = await client.messages.create(
    messages=[...],
    top_k=40
)
```

#### OpenAI
```python
# Response format
response = await client.messages.create(
    messages=[...],
    response_format={"type": "json_object"}
)

# Frequency penalty
response = await client.messages.create(
    messages=[...],
    frequency_penalty=0.5,
    presence_penalty=0.5
)

# Seed for reproducibility
response = await client.messages.create(
    messages=[...],
    seed=12345
)
```

#### Google/Vertex AI
```python
# Safety settings
response = await client.messages.create(
    messages=[...],
    safety_settings=[
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        }
    ]
)

# Candidate count
response = await client.messages.create(
    messages=[...],
    candidate_count=3
)
```

### Why This Design?

1. **Flexibility**: New provider features can be used immediately without updating v-router
2. **Forward Compatibility**: Providers can add new parameters without breaking existing code
3. **Clean Separation**: Core routing logic is separate from provider-specific features
4. **Type Safety**: Core parameters remain type-checked while allowing provider flexibility

### Examples with Fallback Models

Provider-specific parameters work seamlessly with fallback models:

```python
# Primary model with provider-specific params
llm_config = LLM(
    model_name="claude-opus-4-20250514",
    provider="anthropic",
    max_tokens=10000,
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        )
    ]
)

client = Client(llm_config)

# Provider-specific params are passed to whichever model handles the request
response = await client.messages.create(
    messages=[{"role": "user", "content": "Complex analysis"}],
    # These params will be used by Anthropic if available,
    # ignored by OpenAI if it falls back
    timeout=600,
    thinking={"type": "enabled", "budget_tokens": 5000},
    # These params will be used by OpenAI if it falls back,
    # ignored by Anthropic
    seed=12345,
    frequency_penalty=0.2
)
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