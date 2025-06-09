# Configuration

v-router requires API keys for the providers you want to use. This guide covers how to set up authentication and configure various options.

## Environment Variables

Set up authentication for your providers using environment variables:

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

Get your API key from [Anthropic Console](https://console.anthropic.com/).

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-key-here"
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### Google AI Studio

```bash
export GOOGLE_API_KEY="your-google-ai-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Google Cloud (Vertex AI)

For Vertex AI, you need service account credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"  # Optional, defaults to us-central1
```

Set up service account:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Vertex AI API
3. Create a service account with Vertex AI permissions
4. Download the JSON key file
5. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY="your-azure-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

Get your credentials from [Azure Portal](https://portal.azure.com/).

## Configuration File

You can also use a `.env` file in your project root:

```bash title=".env"
# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key-here

# OpenAI
OPENAI_API_KEY=your-openai-key-here

# Google AI
GOOGLE_API_KEY=your-google-ai-key-here

# Google Cloud (Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

v-router automatically loads `.env` files using `python-dotenv`.

## LLM Configuration Options

The `LLM` class supports many configuration options:

### Basic Configuration

```python
from v_router import LLM

llm_config = LLM(
    model_name="claude-sonnet-4",     # Model to use
    provider="anthropic",             # Provider to use
    max_tokens=1000,                  # Max tokens to generate
    temperature=0.7,                  # Creativity (0-1)
    system_prompt="You are helpful"   # System message
)
```

### Advanced Configuration

```python
from v_router import LLM, BackupModel

llm_config = LLM(
    # Core settings
    model_name="claude-sonnet-4",
    provider="anthropic",
    
    # Generation parameters
    max_tokens=2000,
    temperature=0.8,
    top_p=0.9,
    top_k=40,
    
    # System configuration
    system_prompt="You are an expert Python developer",
    
    # Backup models
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        ),
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2
        )
    ],
    
    # Cross-provider fallback
    try_other_providers=True,
    
    # Tool configuration
    tools=None,  # Will be covered in function calling guide
    
    # Provider-specific options
    anthropic_kwargs={"stream": False},
    openai_kwargs={"presence_penalty": 0.1},
    google_kwargs={"safety_settings": {}},
)
```

## Model Name Mapping

v-router automatically maps generic model names to provider-specific versions using the `models.yml` configuration:

### Available Models

=== "Claude Models"

    ```python
    # These all work across Anthropic and Vertex AI
    LLM(model_name="claude-3-opus", provider="anthropic")
    LLM(model_name="claude-3-sonnet", provider="anthropic") 
    LLM(model_name="claude-3-5-haiku", provider="anthropic")
    LLM(model_name="claude-sonnet-4", provider="anthropic")
    LLM(model_name="claude-opus-4", provider="anthropic")
    
    # Same models on Vertex AI
    LLM(model_name="claude-sonnet-4", provider="vertexai")
    ```

=== "GPT Models"

    ```python
    # OpenAI and Azure OpenAI
    LLM(model_name="gpt-4", provider="openai")
    LLM(model_name="gpt-4-turbo", provider="openai")
    LLM(model_name="gpt-4.1", provider="openai")
    LLM(model_name="gpt-3.5", provider="openai")
    
    # Same models on Azure
    LLM(model_name="gpt-4", provider="azure")
    ```

=== "Gemini Models"

    ```python
    # Google AI and Vertex AI
    LLM(model_name="gemini-pro", provider="google")
    LLM(model_name="gemini-1.5-pro", provider="google")
    LLM(model_name="gemini-1.5-flash", provider="google")
    LLM(model_name="gemini-2.0-flash", provider="google")
    
    # Same models on Vertex AI
    LLM(model_name="gemini-1.5-pro", provider="vertexai")
    ```

### Custom Model Names

You can also use provider-specific model names directly:

```python
# Use exact provider model names
LLM(model_name="claude-sonnet-4-20250514", provider="anthropic")
LLM(model_name="gpt-4-turbo-preview", provider="openai")
LLM(model_name="gemini-1.5-pro-latest", provider="google")
```

## Provider-Specific Parameters

v-router supports provider-specific parameters through a two-tier system:

1. **Core parameters** (in LLM class): Used for routing and fallback logic
2. **Provider-specific parameters** (via kwargs): Passed directly to providers

### How to Pass Provider-Specific Parameters

Provider-specific parameters should be passed when creating messages, NOT in the LLM configuration:

```python
from v_router import Client, LLM

# Configure core parameters
client = Client(
    llm_config=LLM(
        model_name="claude-opus-4-20250514",
        provider="anthropic",
        max_tokens=32000,
        temperature=1
    )
)

# Pass provider-specific parameters at message creation
response = await client.messages.create(
    messages=[{"role": "user", "content": "Solve this complex problem"}],
    # Provider-specific parameters:
    timeout=600,  # Anthropic: extended timeout
    thinking={    # Anthropic: thinking mode
        "type": "enabled",
        "budget_tokens": 10000
    }
)
```

### Common Provider-Specific Parameters

#### Anthropic
```python
# Extended timeout for long-running requests
response = await client.messages.create(
    messages=[...],
    timeout=600  # 10 minutes
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
# JSON mode
response = await client.messages.create(
    messages=[...],
    response_format={"type": "json_object"}
)

# Frequency and presence penalties
response = await client.messages.create(
    messages=[...],
    frequency_penalty=0.5,
    presence_penalty=0.5
)

# Reproducible outputs
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

# Multiple candidates
response = await client.messages.create(
    messages=[...],
    candidate_count=3
)
```

### With Fallback Models

Provider-specific parameters work seamlessly with fallbacks:

```python
llm_config = LLM(
    model_name="claude-opus-4-20250514",
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4o", provider="openai"),
            priority=1
        )
    ]
)

client = Client(llm_config)

# These params are passed to whichever provider handles the request
response = await client.messages.create(
    messages=[{"role": "user", "content": "Analyze this"}],
    # Anthropic params (used if primary succeeds)
    timeout=600,
    thinking={"type": "enabled"},
    # OpenAI params (used if fallback occurs)
    seed=12345,
    frequency_penalty=0.2
)
```

## Provider-Specific Configuration

### Anthropic Configuration

```python
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    anthropic_kwargs={
        "stream": False,
        "extra_headers": {"Custom-Header": "value"}
    }
)
```

### OpenAI Configuration

```python
llm_config = LLM(
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

### Google Configuration

```python
llm_config = LLM(
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

### Vertex AI Configuration

```python
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="vertexai",
    vertex_kwargs={
        "location": "us-central1",
        "project": "my-project-id"
    }
)
```

## Backup Model Configuration

Configure multiple fallback strategies:

```python
from v_router import LLM, BackupModel

llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    
    # Backup models in priority order
    backup_models=[
        # Try OpenAI GPT-4 first
        BackupModel(
            model=LLM(
                model_name="gpt-4o",
                provider="openai",
                max_tokens=1000,
                temperature=0.7
            ),
            priority=1
        ),
        # Then try Google Gemini
        BackupModel(
            model=LLM(
                model_name="gemini-1.5-pro",
                provider="google",
                max_tokens=1000,
                temperature=0.7
            ),
            priority=2
        ),
        # Finally try Azure OpenAI
        BackupModel(
            model=LLM(
                model_name="gpt-4",
                provider="azure",
                max_tokens=1000,
                temperature=0.7
            ),
            priority=3
        )
    ],
    
    # Also try the same model on other providers
    try_other_providers=True
)
```

## Logging Configuration

Enable detailed logging to debug issues:

```python
import logging
from v_router import setup_logger

# Enable debug logging
setup_logger("v_router", level=logging.DEBUG)

# Or configure specific loggers
logging.getLogger("v_router.router").setLevel(logging.INFO)
logging.getLogger("v_router.providers").setLevel(logging.DEBUG)
```

## Validation

v-router validates your configuration and provides helpful error messages:

```python
from v_router import LLM

try:
    # This will raise an error if the API key is missing
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic"
    )
    client = Client(llm_config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Testing Configuration

Test your configuration with a simple request:

```python
import asyncio
from v_router import Client, LLM

async def test_config():
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic"
    )
    
    client = Client(llm_config)
    
    try:
        response = await client.messages.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"✅ Configuration working! Model: {response.model}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

asyncio.run(test_config())
```

## Best Practices

### Security

- **Never commit API keys** to version control
- Use environment variables or secure secret management
- Rotate API keys regularly
- Monitor API usage and costs

### Performance

- Configure appropriate `max_tokens` limits
- Use backup models to ensure availability
- Consider using faster models for development

### Cost Management

- Monitor token usage with `response.usage`
- Use cheaper models for backup when appropriate
- Set `max_tokens` to control costs

## Next Steps

Now that you have v-router configured:

1. [Try the quick start guide](quick-start.md)
2. [Learn about function calling](../guide/function-calling.md)
3. [Explore multimodal content](../guide/multimodal-content.md)

[Quick Start Guide →](quick-start.md){ .md-button .md-button--primary }
[Function Calling →](../guide/function-calling.md){ .md-button }