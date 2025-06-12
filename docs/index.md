# v-router

A unified LLM interface that provides automatic fallback between different LLM providers. Route your AI requests seamlessly across **Anthropic**, **OpenAI**, **Google**, and **Azure** with intelligent failover strategies and a consistent API.

## ✨ Key Features

- **🚀 Automatic Fallback**: Seamlessly switch between models and providers when failures occur. Configure backup models and cross-provider fallback strategies. [Learn more →](getting-started/quick-start.md)

- **🔗 Unified API**: Same interface works across all major LLM providers. Write once, run anywhere with consistent request/response formats. [Explore the API →](api/client.md)

- **⚡ Smart Routing**: Intelligent model selection based on availability and configuration. Automatic model name mapping across providers. [See provider configuration →](getting-started/configuration.md)

- **🛠️ Function Calling**: Unified tool calling interface across all providers. Use the same function definitions everywhere. Control tool usage with force, disable, or auto modes. [Function calling guide →](guide/function-calling.md)

- **🖼️ Multimodal Support**: Send images, PDFs, and Word documents with automatic format conversion. Support for vision models across all providers. [Multimodal examples →](getting-started/quick-start.md)

- **⚙️ Flexible Configuration**: Fine-tune parameters, backup models, and provider priorities. Extensive customization options. [Configuration guide →](getting-started/configuration.md)

## Quick Example

```python
from v_router import Client, LLM

# Create an LLM configuration with automatic fallback
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        {"model": LLM(model_name="gpt-4o", provider="openai"), "priority": 1},
        {"model": LLM(model_name="gemini-1.5-pro", provider="google"), "priority": 2}
    ]
)

# Create a client
client = Client(llm_config)

# Send a message - automatically falls back if primary model fails
response = await client.messages.create(
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
)

print(f"Response: {response.content[0].text}")
print(f"Model used: {response.model} ({response.provider})")
```

## Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku), Claude 4 (Opus, Sonnet) | Function calling, Images, PDFs, Word docs |
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-4.1, GPT-3.5 | Function calling, Images, PDFs, Word docs |
| **Google AI** | Gemini Pro, Gemini 1.5 (Pro, Flash), Gemini 2.0 Flash | Function calling, Images, PDFs, Word docs |
| **Azure OpenAI** | GPT-4, GPT-4 Turbo, GPT-4.1, GPT-3.5 | Function calling, Images, PDFs, Word docs |
| **Vertex AI** | Claude 3/4 & Gemini models via Google Cloud | Function calling, Images, PDFs, Word docs |

## Why v-router?

### Reliability Through Redundancy
Never worry about API outages or rate limits again. v-router automatically routes requests to backup models and alternative providers when failures occur.

### Simplified Integration
Write your code once and deploy it anywhere. The unified interface abstracts away provider-specific differences while maintaining full feature parity.

### Cost Optimization
Easily switch between providers based on cost, performance, or availability. Configure fallback strategies that optimize for your specific use case.

### Future-Proof Architecture
Add new providers and models without changing your application code. The modular architecture makes it easy to extend and customize.

## Getting Started

Ready to get started? Follow our quick start guide:

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View Examples](examples/basic.md){ .md-button }

## Development Roadmap

- [x] **Chat Completions**: Unified interface across providers 
- [x] **Function Calling**: Tool calling support with force/disable/auto modes
- [x] **Multimodal Support**: Images, PDFs, and Word document processing
- [ ] **Streaming**: Real-time response streaming
- [ ] **AWS Bedrock**: Additional provider support
- [ ] **JSON Mode**: Structured output generation
- [ ] **Prompt Caching**: Optimization for repeated prompts
- [ ] **Ollama Support**: Local model integration

## Community

- **Repository**: [GitHub](https://github.com/vectrix-ai/v-router)
- **Package**: [PyPI](https://pypi.org/project/v-router/)
- **Issues**: [GitHub Issues](https://github.com/vectrix-ai/v-router/issues)
- **Email**: [ben@vectrix.ai](mailto:ben@vectrix.ai)

---

**v-router** - Making LLM integration simple, reliable, and unified across all providers.