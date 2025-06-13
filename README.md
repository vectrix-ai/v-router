# v-router

A unified LLM interface that provides automatic fallback between different LLM providers. Route your AI requests seamlessly across **Anthropic**, **OpenAI**, **Google**, and **Azure** with intelligent failover strategies and a consistent API.

## ✨ Features

- **🚀 Automatic Fallback**: Seamless switching between models and providers when failures occur
- **📚 Unified API**: Same interface works across all major LLM providers
- **⚡ Smart Routing**: Intelligent model selection based on availability and configuration
- **🔧 Function Calling**: Unified tool calling interface across all providers
- **🖼️ Multimodal Support**: Send images and PDFs with automatic format conversion
- **🎯 Consistent Responses**: Standardized response format regardless of provider
- **⚙️ Flexible Configuration**: Fine-tune parameters, backup models, and provider priorities

## 📦 Installation

```bash
pip install v-router
```

Or with development dependencies:

```bash
uv sync --all-extras
```

## 🚀 Quick Start

### Basic Conversation

```python
from v_router import Client, LLM
from v_router import HumanMessage, AIMessage, SystemMessage

# Initialize client
client = Client(
    llm_config=LLM(
        model_name="claude-3-sonnet-20240229",
        provider="anthropic",
        max_tokens=500
    )
)

# Basic conversation
response = await client.messages.create(
    messages=[
        SystemMessage("You are a helpful assistant"),
        HumanMessage("What's the capital of France?")
    ]
)

print(response.content)  # "The capital of France is Paris."
```

### Response Format

All providers return a consistent `AIMessage` format:

```python
# The response object
print(response.content)        # "The capital of France is Paris." (or ["Part 1", "Part 2"] for multi-part)
print(response.model)          # "claude-3-sonnet-20240229"
print(response.provider)       # "anthropic"
print(response.usage.input_tokens)   # 25
print(response.usage.output_tokens)  # 8
```

### Multi-turn Conversations

```python
# Start a conversation
messages = [
    SystemMessage("You are a helpful math tutor"),
    HumanMessage("What's 15% of 80?")
]

response = await client.messages.create(messages=messages)
print(response.content)  # "To find 15% of 80..."

# Continue the conversation - just append the response
messages.append(response)  # AIMessage is automatically handled
messages.append(HumanMessage("Can you show me another way to calculate it?"))

response2 = await client.messages.create(messages=messages)
print(response2.content)  # "Sure! Another way to calculate 15% of 80..."
```

## 🔧 Function Calling (Tool Use)

```python
from v_router import Client, LLM, HumanMessage, ToolMessage
from v_router.classes.tools import ToolCall, Tools
from pydantic import BaseModel, Field

# Define your tool
class Calculator(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

calculator_tool = ToolCall(
    name="calculator",
    description="Evaluate mathematical expressions",
    input_schema=Calculator.model_json_schema()
)

# Create client with tools
client = Client(
    llm_config=LLM(
        model_name="gpt-4",
        provider="openai",
        tools=Tools(tools=[calculator_tool])
    )
)

# Ask a question that requires the tool
messages = [HumanMessage("What's 392 * 47?")]
response = await client.messages.create(messages=messages)

# Check if the model wants to use a tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Tool: {tool_call.name}")           # "calculator"
    print(f"Arguments: {tool_call.args}")      # {"expression": "392 * 47"}
    
    # Execute the tool (your implementation)
    result = eval(tool_call.args["expression"])  # 18,424
    
    # Send tool result back
    messages.append(response)  # Add the AIMessage with tool_calls
    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call.id
        )
    )
    
    # Get final response
    final_response = await client.messages.create(messages=messages)
    print(final_response.content)  # "392 * 47 = 18,424"
```

## 🖼️ Multimodal Messages

### Sending Images

```python
from v_router import Client, LLM, HumanMessage
from v_router.classes.messages import TextContent, ImageContent
import base64

client = Client(
    llm_config=LLM(
        model_name="gpt-4o",
        provider="openai"
    )
)

# Method 1: Direct file path (auto-converted to base64)
response = await client.messages.create(
    messages=[
        HumanMessage("image.jpg"),  # Just pass the file path
        HumanMessage("What's in this image?")
    ]
)

# Method 2: Explicit multimodal content
with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = await client.messages.create(
    messages=[
        HumanMessage([
            TextContent(text="Analyze this chart and tell me the trend:"),
            ImageContent(data=image_data, media_type="image/png")
        ])
    ]
)

print(response.content)  # "The chart shows an upward trend..."
```

### Working with Documents (PDFs)

```python
from v_router.classes.messages import DocumentContent

# PDFs work similarly to images
with open("report.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

response = await client.messages.create(
    messages=[
        HumanMessage([
            TextContent(text="Summarize this report:"),
            DocumentContent(
                data=pdf_data,
                media_type="application/pdf"
            )
        ])
    ]
)

# Or just use the file path
response = await client.messages.create(
    messages=[
        HumanMessage("report.pdf"),  # Auto-converted
        HumanMessage("What are the key findings?")
    ]
)
```

## 🚀 Automatic Fallback

Configure backup models to ensure reliability:

```python
from v_router import Client, LLM, BackupModel

llm_config = LLM(
    model_name="claude-3-opus-20240229",  # Primary model
    provider="anthropic",
    backup_models=[
        BackupModel(
            model=LLM(model_name="gpt-4", provider="openai"),
            priority=1  # First fallback
        ),
        BackupModel(
            model=LLM(model_name="gemini-1.5-pro", provider="google"),
            priority=2  # Second fallback
        )
    ]
)

client = Client(llm_config)

# If Claude fails, automatically tries GPT-4, then Gemini
response = await client.messages.create(
    messages=[HumanMessage("Hello!")]
)

print(f"Response from: {response.provider}")  # Shows which provider succeeded
```

## 🔄 Cross-Provider Switching

Enable automatic cross-provider fallback for the same model:

```python
llm_config = LLM(
    model_name="claude-3-sonnet-20240229",
    provider="vertexai",  # Try Vertex AI first
    try_other_providers=True  # Fall back to direct Anthropic if needed
)

# If Vertex AI fails, automatically tries the same model on Anthropic
response = await client.messages.create(
    messages=[HumanMessage("Hello!")]
)
```

## 🎯 Provider-Specific Parameters

v-router supports provider-specific features through flexible parameters:

```python
# Anthropic - Thinking mode
response = await client.messages.create(
    messages=[HumanMessage("Solve this complex problem: ...")],
    timeout=600,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    }
)

# OpenAI - JSON mode
response = await client.messages.create(
    messages=[HumanMessage("Return a JSON object with name and age")],
    response_format={"type": "json_object"},
    seed=12345
)

# Google - Safety settings
response = await client.messages.create(
    messages=[HumanMessage("Tell me about...")],
    safety_settings=[{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
    }]
)
```

## 🌐 Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku), Claude 4 (Opus, Sonnet) | Function calling, Images, PDFs |
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 | Function calling, Images |
| **Google** | Gemini Pro, Gemini 1.5 (Pro, Flash), Gemini 2.0 Flash | Function calling, Images, PDFs |
| **Azure OpenAI** | GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 | Function calling, Images |
| **Vertex AI** | Claude 3/4 & Gemini models via Google Cloud | Function calling, Images, PDFs |

## ⚙️ Configuration

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

# LangFuse (optional - for tracing)
export LANGFUSE_HOST="your-langfuse-host"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
```

### 📊 LangFuse Tracing Support

v-router includes built-in support for [LangFuse](https://langfuse.com/) tracing:

```
📊 LLM Request
├── 🔴 Primary Model (claude-3-opus on anthropic) - FAILED
│   ├── Error: Rate limit exceeded
│   └── Duration: 150ms
└── ✅ Backup Model (gpt-4 on openai) - SUCCESS
    ├── Input tokens: 45
    ├── Output tokens: 128
    ├── Duration: 2.3s
    └── Cost: $0.0043
```

Enable by setting the `LANGFUSE_HOST` environment variable. No code changes required.

### Model Configuration

v-router uses `models.yml` to map model names across providers:

```python
# These all work automatically:
LLM(model_name="claude-3-sonnet", provider="anthropic")      # → claude-3-sonnet-20240229
LLM(model_name="claude-3-sonnet", provider="vertexai")       # → claude-3-sonnet@20240229
LLM(model_name="gpt-4", provider="openai")                   # → gpt-4
LLM(model_name="gemini-1.5-pro", provider="google")          # → gemini-1.5-pro-latest
```

## 📖 Documentation

Complete documentation is available online:

**📚 [Full Documentation](https://vectrix-ai.github.io/v-router/)**

### Jupyter Notebooks

Explore the [`examples/`](examples/) directory for interactive examples:

- **[quickstart_models.ipynb](examples/quickstart_models.ipynb)**: Basic usage, fallbacks, and cross-provider switching
- **[quickstart_tool_calling.ipynb](examples/quickstart_tool_calling.ipynb)**: Function calling across providers
- **[multimodal_content.ipynb](examples/multimodal_content.ipynb)**: Working with images and PDFs

## 🛠️ Development

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

## 🏗️ Architecture

v-router follows a clean provider pattern:

- **Client**: Main entry point with unified API
- **Router**: Handles request routing and fallback logic
- **Providers**: Individual provider implementations
- **Models**: Unified request/response models

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/vectrix-ai/v-router/issues)
- **Documentation**: See examples in the [`examples/`](examples/) directory
- **Email**: [ben@vectrix.ai](mailto:ben@vectrix.ai)

---

**v-router** - Making LLM integration simple, reliable, and unified across all providers.