# v-router

A unified LLM interface that provides automatic fallback between different LLM providers. Route your AI requests seamlessly across **Anthropic**, **OpenAI**, **Google**, and **Azure** with intelligent failover strategies and a consistent API.

## =� Features

- **= Automatic Fallback**: Seamless switching between models and providers when failures occur
- **<� Unified API**: Same interface works across all major LLM providers
- **� Smart Routing**: Intelligent model selection based on availability and configuration
- **=� Function Calling**: Unified tool calling interface across all providers
- **=� Consistent Responses**: Standardized response format regardless of provider
- **=' Flexible Configuration**: Fine-tune parameters, backup models, and provider priorities

## =� Installation

```bash
pip install v-router
```

Or with development dependencies:

```bash
uv sync --all-extras
```

## <� Quick Start

### Basic Usage

```python
from v_router import Client, LLM

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
```

### Automatic Fallback

Configure backup models to ensure reliability:

```python
from v_router import Client, LLM, BackupModel

llm_config = LLM(
    model_name="claude-6",  # Primary model (might fail)
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

# If claude-6 fails, automatically tries gpt-4o, then gemini-1.5-pro
response = await client.messages.create(
    messages=[{"role": "user", "content": "What's 2+2?"}]
)
```

### Cross-Provider Switching

Enable automatic cross-provider fallback for the same model:

```python
llm_config = LLM(
    model_name="claude-opus-4",
    provider="vertexai",  # Try Vertex AI first
    try_other_providers=True  # Fall back to direct Anthropic if needed
)

client = Client(llm_config)
response = await client.messages.create(
    messages=[{"role": "user", "content": "Tell me a joke."}]
)
```

## =� Function Calling

v-router provides unified function calling across all providers:

```python
from pydantic import BaseModel, Field
from v_router import Client, LLM
from v_router.classes.tools import ToolCall, Tools

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

# Configure LLM with tools
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=Tools(tools=[weather_tool])
)

client = Client(llm_config)

# Make request
response = await client.messages.create(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

# Check for tool calls
if response.tool_use:
    for tool_call in response.tool_use:
        print(f"Tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
```

## =� Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku), Claude 4 (Opus, Sonnet) | Function calling |
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-4.1, GPT-3.5 | Function calling |
| **Google** | Gemini Pro, Gemini 1.5 (Pro, Flash), Gemini 2.0 Flash | Function calling |
| **Azure OpenAI** | GPT-4, GPT-4 Turbo, GPT-4.1, GPT-3.5 | Function calling |
| **Vertex AI** | Claude 3/4 & Gemini models via Google Cloud | Function calling |

## Configuration

### Environment Variables

Set up authentication for your providers:

```bash
# Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# OpenAI
export OPENAI_API_KEY="your-key-here"

# Google AI Studio
export GOOGLE_API_KEY="your-key-here"

# Google Cloud (for Vertex AI)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

### Model Configuration

v-router uses `models.yml` to map model names across providers. You can use generic names that automatically map to provider-specific models:

```python
# These all work automatically:
LLM(model_name="claude-sonnet-4", provider="anthropic")      # � claude-sonnet-4-20250514
LLM(model_name="claude-sonnet-4", provider="vertexai")       # � claude-sonnet-4@20250514
LLM(model_name="gpt-4", provider="openai")                   # � gpt-4
LLM(model_name="gemini-1.5-pro", provider="google")          # � gemini-1.5-pro-latest
```

## = Response Format

All providers return the same unified response structure:

```python
class Response:
    content: List[Content]          # Text content blocks
    tool_use: List[ToolUse]        # Function calls made
    usage: Usage                   # Token usage info  
    model: str                     # Actual model used
    provider: str                  # Provider used
    raw_response: Any              # Original provider response
```

## =� Examples

Explore the [`examples/`](examples/) directory for detailed usage:

- **[quickstart_models.ipynb](examples/quickstart_models.ipynb)**: Basic usage, fallbacks, and cross-provider switching
- **[quickstart_tool_calling.ipynb](examples/quickstart_tool_calling.ipynb)**: Function calling across providers
- **[providers/](examples/providers/)**: Provider-specific examples

## =� Development Roadmap

- [x] **Chat Completions**: Unified interface across providers 
- [x] **Function Calling**: Tool calling support 
- [ ] **Multimodal Support**: Images, PDFs, and document processing
- [ ] **Streaming**: Real-time response streaming
- [ ] **LangSmith Integration**: Advanced monitoring and tracing
- [ ] **AWS Bedrock**: Additional provider support
- [ ] **JSON Mode**: Structured output generation
- [ ] **Prompt Caching**: Optimization for repeated prompts
- [ ] **Ollama Support**: Local model integration

## >� Development

### Setup

```bash
# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/models/test_llm.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## <� Architecture

v-router follows a clean provider pattern:

- **Client**: Main entry point with unified API
- **Router**: Handles request routing and fallback logic
- **Providers**: Individual provider implementations inheriting from `BaseProvider`
- **Models**: Unified request/response models

### Adding a New Provider

1. Create provider class in `src/v_router/providers/`
2. Inherit from `BaseProvider`
3. Implement `create_message()` and `name` property
4. Add to `PROVIDER_REGISTRY` in `router.py`
5. Update `models.yml` with supported models

## =� License

This project is licensed under the MIT License.

## > Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## =� Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Documentation**: See examples in the [`examples/`](examples/) directory
- **Email**: [ben@vectrix.ai](mailto:ben@vectrix.ai)

---

**v-router** - Making LLM integration simple, reliable, and unified across all providers.