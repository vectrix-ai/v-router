# LLM Configuration

The `LLM` class defines the configuration for Large Language Models in v-router. It specifies which model to use, the provider, generation parameters, and fallback strategies.

## LLM Class

```python
from v_router import LLM

llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    max_tokens=1000,
    temperature=0.7
)
```

### Parameters

#### Required Parameters

**`model_name`** (str)  
Name of the LLM model to use. v-router automatically maps generic names to provider-specific versions.

Examples:
- `"claude-sonnet-4"` → `"claude-sonnet-4-20250514"` (Anthropic) or `"claude-sonnet-4@20250514"` (Vertex AI)
- `"gpt-4o"` → `"gpt-4o"` (OpenAI/Azure)
- `"gemini-1.5-pro"` → `"gemini-1.5-pro-latest"` (Google/Vertex AI)

**`provider`** (Literal["openai", "anthropic", "azure", "google", "vertexai"])  
The LLM provider to use:

- `"openai"`: OpenAI API
- `"anthropic"`: Anthropic API
- `"azure"`: Azure OpenAI Service
- `"google"`: Google AI Studio
- `"vertexai"`: Google Cloud Vertex AI (supports both Claude and Gemini models)

#### Optional Parameters

**`max_tokens`** (Optional[int] = None)  
Maximum number of tokens to generate. If not specified, uses the model's default maximum.

**`temperature`** (Optional[float] = 0)  
Sampling temperature (0-1). Higher values make output more random, lower values more deterministic.

**`backup_models`** (List[BackupModel] = [])  
List of backup models to try if the primary model fails. See [BackupModel](#backupmodel-class) below.

**`try_other_providers`** (bool = False)  
If True, automatically tries the same model on other providers if the primary provider fails.

**`tools`** (Optional[Tools] = None)  
Function calling tools available to the model. See [Tools documentation](tools.md).

**`tool_choice`** (Optional[Union[str, Dict[str, Any]]] = None)  
Controls how the model uses tools. Options:

- `None` or `"auto"`: Model decides whether to use tools (default)
- `"any"`: Model must use one of the provided tools
- `"none"`: Model is prevented from using tools
- `str` (tool name): Force the model to use a specific tool
- `dict`: Provider-specific format for advanced control

### Provider-Specific Parameters

**`anthropic_kwargs`** (dict = {})  
Additional parameters passed to the Anthropic provider:

```python
LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    anthropic_kwargs={
        "stream": False,
        "extra_headers": {"Custom-Header": "value"}
    }
)
```

**`openai_kwargs`** (dict = {})  
Additional parameters passed to OpenAI provider:

```python
LLM(
    model_name="gpt-4o",
    provider="openai",
    openai_kwargs={
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "logit_bias": {},
        "user": "user-123"
    }
)
```

**`google_kwargs`** (dict = {})  
Additional parameters passed to Google providers:

```python
LLM(
    model_name="gemini-1.5-pro",
    provider="google",
    google_kwargs={
        "safety_settings": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE"
        },
        "generation_config": {
            "candidate_count": 1
        }
    }
)
```

## BackupModel Class

Defines a backup model with priority ordering:

```python
from v_router import BackupModel, LLM

backup = BackupModel(
    model=LLM(model_name="gpt-4o", provider="openai"),
    priority=1  # Lower numbers = higher priority
)
```

### Parameters

**`model`** (LLM)  
The backup LLM configuration to use.

**`priority`** (int)  
Priority level (1 = highest priority). Must be unique across all backup models.

## Usage Examples

### Basic Configuration

```python
from v_router import LLM

# Simple configuration
basic_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic"
)

# With generation parameters
tuned_config = LLM(
    model_name="gpt-4o",
    provider="openai",
    max_tokens=2000,
    temperature=0.8,
    system_prompt="You are a creative writing assistant"
)
```

### Fallback Configuration

```python
from v_router import LLM, BackupModel

# Multiple backup models
robust_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        ),
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2
        ),
        BackupModel(
            model=LLM(model_name="gpt-4", provider="azure"),
            priority=3
        )
    ],
    try_other_providers=True  # Also try same model on other providers
)
```

### Cross-Provider Configuration

```python
# Try the same model across multiple providers
cross_provider_config = LLM(
    model_name="claude-sonnet-4",
    provider="vertexai",  # Try Vertex AI first
    try_other_providers=True,  # Fall back to Anthropic if needed
    max_tokens=1000
)
```

### Function Calling Configuration

```python
from v_router.classes.tools import ToolCall, Tools
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    location: str = Field(..., description="City name")

weather_tool = ToolCall(
    name="get_weather",
    description="Get weather information",
    input_schema=WeatherQuery.model_json_schema()
)

function_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=Tools(tools=[weather_tool]),
    tool_choice="auto"  # Optional: control tool usage
)
```

### Tool Choice Configuration

Control when and how tools are used:

```python
# Force a specific tool
force_weather = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic", 
    tools=Tools(tools=[weather_tool, calculator_tool]),
    tool_choice="get_weather"  # Force weather tool
)

# Require any tool usage
require_tools = LLM(
    model_name="gpt-4o",
    provider="openai",
    tools=Tools(tools=[weather_tool, calculator_tool]),
    tool_choice="any"  # Must use one of the tools
)

# Disable tool usage
no_tools = LLM(
    model_name="gemini-1.5-pro",
    provider="google",
    tools=Tools(tools=[weather_tool, calculator_tool]),
    tool_choice="none"  # Prevent tool usage
)

# Provider-specific format (Anthropic)
anthropic_specific = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=Tools(tools=[calculator_tool]),
    tool_choice={"type": "tool", "name": "calculator"}
)

# Provider-specific format (OpenAI)
openai_specific = LLM(
    model_name="gpt-4o",
    provider="openai", 
    tools=Tools(tools=[calculator_tool]),
    tool_choice={"type": "function", "function": {"name": "calculator"}}
)
```

## Model Name Mapping

v-router automatically maps generic model names to provider-specific versions based on `models.yml`:

### Anthropic Models

| Generic Name | Anthropic API | Vertex AI |
|--------------|---------------|-----------|
| `claude-3-opus` | `claude-3-opus-20240229` | `claude-3-opus@20240229` |
| `claude-3-sonnet` | `claude-3-sonnet-20240229` | `claude-3-sonnet@20240229` |
| `claude-3-haiku` | `claude-3-haiku-20240307` | `claude-3-haiku@20240307` |
| `claude-sonnet-4` | `claude-sonnet-4-20250514` | `claude-sonnet-4@20250514` |
| `claude-opus-4` | `claude-opus-4-20250514` | `claude-opus-4@20250514` |

### OpenAI Models

| Generic Name | OpenAI/Azure API |
|--------------|------------------|
| `gpt-4` | `gpt-4` |
| `gpt-4-turbo` | `gpt-4-turbo-preview` |
| `gpt-4.1` | `gpt-4.1` |
| `gpt-3.5` | `gpt-3.5-turbo` |

### Google Models

| Generic Name | Google AI/Vertex API |
|--------------|----------------------|
| `gemini-pro` | `gemini-1.0-pro` |
| `gemini-1.5-pro` | `gemini-1.5-pro-latest` |
| `gemini-1.5-flash` | `gemini-1.5-flash-latest` |
| `gemini-2.0-flash` | `gemini-2.0-flash-001` |

## Validation

The LLM class includes validation for configuration parameters:

### Priority Validation

```python
# ❌ This will raise an error (duplicate priorities)
LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(model=LLM(model_name="gpt-4o", provider="openai"), priority=1),
        BackupModel(model=LLM(model_name="gemini-1.5-pro", provider="google"), priority=1)  # Duplicate!
    ]
)

# ✅ This is correct (unique priorities)
LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(model=LLM(model_name="gpt-4o", provider="openai"), priority=1),
        BackupModel(model=LLM(model_name="gemini-1.5-pro", provider="google"), priority=2)
    ]
)
```

### Parameter Validation

```python
# ❌ Invalid priority (must be >= 1)
BackupModel(
    model=LLM(model_name="gpt-4o", provider="openai"),
    priority=0  # Error!
)

# ❌ Invalid temperature (must be 0-1)
LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    temperature=2.0  # Error!
)
```

## Methods

### get_ordered_backup_models()

Returns backup models sorted by priority (lowest number first):

```python
config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(model=LLM(model_name="gpt-4o", provider="openai"), priority=2),
        BackupModel(model=LLM(model_name="gemini-1.5-pro", provider="google"), priority=1)
    ]
)

ordered_backups = config.get_ordered_backup_models()
# Returns: [gemini-1.5-pro (priority 1), gpt-4o (priority 2)]
```

## Advanced Patterns

### Environment-Specific Configuration

```python
import os

def get_llm_config(environment: str) -> LLM:
    if environment == "production":
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=2000,
            temperature=0.3,  # More deterministic for production
            backup_models=[
                BackupModel(
                    model=LLM(model_name="gpt-4o", provider="openai"),
                    priority=1
                )
            ]
        )
    else:
        return LLM(
            model_name="gpt-3.5",  # Cheaper for development
            provider="openai",
            max_tokens=500,
            temperature=0.7
        )

config = get_llm_config(os.getenv("ENVIRONMENT", "development"))
```

### Dynamic Model Selection

```python
def get_model_for_task(task_type: str) -> LLM:
    if task_type == "creative_writing":
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            temperature=0.9,  # High creativity
            max_tokens=3000
        )
    elif task_type == "code_analysis":
        return LLM(
            model_name="gpt-4o",
            provider="openai",
            temperature=0.1,  # Low creativity, high precision
            max_tokens=2000
        )
    elif task_type == "data_analysis":
        return LLM(
            model_name="gemini-1.5-pro",
            provider="google",
            temperature=0.3,
            max_tokens=4000
        )
    else:
        # Default configuration
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic"
        )
```

## Best Practices

### Configuration Management

1. **Use backup models** for production systems
2. **Set appropriate max_tokens** to control costs
3. **Tune temperature** based on use case
4. **Enable cross-provider fallback** for critical applications

### Security

1. **Never hardcode API keys** in LLM configurations
2. **Use environment variables** for sensitive parameters
3. **Validate all inputs** before creating configurations

### Performance

1. **Choose appropriate models** for your use case
2. **Use faster models** for backup when appropriate
3. **Monitor token usage** and adjust limits accordingly

## Related Documentation

- [Client API](client.md): Using LLM configurations with the client
- [Function Calling](../guide/function-calling.md): Adding tools to LLM configurations
- [Provider Configuration](../guide/provider-configuration.md): Provider-specific settings
- [Examples](../examples/basic.md): Real-world configuration examples