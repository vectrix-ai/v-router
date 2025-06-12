# Provider Overview

v-router supports multiple LLM providers, each with their own strengths and capabilities. This page provides an overview of all supported providers and helps you choose the right one for your use case.

## Supported Providers

| Provider | Models | Function Calling | Multimodal | Streaming* | Cost |
|----------|--------|------------------|------------|------------|------|
| **[Anthropic](anthropic.md)** | Claude 3, Claude 4 | ✅ | ✅ Images, PDFs, Word | 🚧 | $$$ |
| **[OpenAI](openai.md)** | GPT-4, GPT-3.5 | ✅ | ✅ Images, Word | 🚧 | $$$ |
| **[Google AI](google.md)** | Gemini 1.5, 2.0 | ✅ | ✅ Images, PDFs, Word | 🚧 | $$ |
| **[Azure OpenAI](azure.md)** | GPT-4, GPT-3.5 | ✅ | ✅ Images, Word | 🚧 | $$$ |
| **[Vertex AI](vertex.md)** | Claude + Gemini | ✅ | ✅ Images, PDFs, Word | 🚧 | $$ |

*Streaming support is planned for future releases

## Provider Comparison

### Anthropic
- **Best for**: High-quality reasoning, analysis, creative writing
- **Models**: Claude 3 (Opus, Sonnet, Haiku), Claude 4 (Opus, Sonnet)
- **Strengths**: Excellent instruction following, safety, long context
- **Use cases**: Complex analysis, creative tasks, code review

### OpenAI
- **Best for**: General-purpose tasks, wide ecosystem support
- **Models**: GPT-4, GPT-4 Turbo, GPT-4.1, GPT-3.5
- **Strengths**: Fast inference, broad capabilities, mature API
- **Use cases**: Chat applications, content generation, code assistance

### Google AI
- **Best for**: Cost-effective multimodal tasks, fast inference
- **Models**: Gemini Pro, Gemini 1.5 (Pro, Flash), Gemini 2.0 Flash
- **Strengths**: Fast, cost-effective, good multimodal support
- **Use cases**: Real-time applications, image analysis, cost-sensitive projects

### Azure OpenAI
- **Best for**: Enterprise deployments, compliance requirements
- **Models**: Same as OpenAI (GPT-4, GPT-3.5)
- **Strengths**: Enterprise features, compliance, data residency
- **Use cases**: Enterprise applications, regulated industries

### Vertex AI
- **Best for**: Google Cloud integration, unified access to multiple models
- **Models**: Claude 3/4 and Gemini models via Google Cloud
- **Strengths**: Single platform for multiple models, enterprise features
- **Use cases**: Google Cloud environments, multi-model workflows

## Model Capabilities

### Text Generation Quality

| Provider | Creative Writing | Technical Analysis | Code Generation |
|----------|------------------|-------------------|-----------------|
| Anthropic Claude | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| OpenAI GPT-4 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Google Gemini | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Multimodal Capabilities

| Provider | Image Understanding | PDF Processing | Word Documents | Document Analysis |
|----------|-------------------|----------------|----------------|-------------------|
| Anthropic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| OpenAI | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Google | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Performance & Cost

| Provider | Latency | Throughput | Cost (Relative) |
|----------|---------|------------|----------------|
| Anthropic | Medium | Medium | High |
| OpenAI | Fast | High | High |
| Google | Very Fast | Very High | Low |
| Azure | Fast | High | High |
| Vertex AI | Medium | High | Medium |

## Choosing a Provider

### For Development & Prototyping
```python
# Start with Google for cost-effectiveness
LLM(model_name="gemini-1.5-pro", provider="google")
```

### For Production Systems
```python
# Use multiple providers with fallback
LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    backup_models=[
        BackupModel(model=LLM(model_name="gpt-4o", provider="openai"), priority=1),
        BackupModel(model=LLM(model_name="gemini-1.5-pro", provider="google"), priority=2)
    ],
    try_other_providers=True
)
```

### For Enterprise
```python
# Azure for compliance or Vertex for Google Cloud
LLM(model_name="gpt-4", provider="azure")
# or
LLM(model_name="claude-sonnet-4", provider="vertexai")
```

## Provider-Specific Features

### Anthropic Features
- Constitutional AI for safety
- Large context windows (up to 200K tokens)
- Excellent at following complex instructions
- Strong performance on reasoning tasks

### OpenAI Features
- Function calling with JSON schemas
- Large ecosystem and community
- Regular model updates
- Good balance of speed and quality

### Google Features
- Very fast inference
- Cost-effective pricing
- Integrated with Google Cloud services
- Strong multimodal capabilities

### Azure Features
- Enterprise-grade security
- Data residency options
- SLA guarantees
- Integration with Microsoft ecosystem

### Vertex AI Features
- Access to multiple model families
- Enterprise security and compliance
- Integrated monitoring and logging
- Custom model training capabilities

## Authentication Setup

Each provider requires different authentication methods:

=== "Anthropic"
    ```bash
    export ANTHROPIC_API_KEY="your-key"
    ```

=== "OpenAI"
    ```bash
    export OPENAI_API_KEY="your-key"
    ```

=== "Google AI"
    ```bash
    export GOOGLE_API_KEY="your-key"
    ```

=== "Azure OpenAI"
    ```bash
    export AZURE_OPENAI_API_KEY="your-key"
    export AZURE_OPENAI_ENDPOINT="your-endpoint"
    ```

=== "Vertex AI"
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
    export GCP_PROJECT_ID="your-project"
    ```

## Cross-Provider Compatibility

v-router ensures consistent behavior across all providers:

### Unified Request Format
```python
# Same request format works with all providers
response = await client.messages.create(
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Unified Response Format
```python
# Same response structure from all providers
print(response.content[0].text)
print(response.model)
print(response.provider)
print(response.usage.total_tokens)
```

### Function Calling Compatibility
```python
# Same tool definition works across providers
tool = ToolCall(
    name="get_weather",
    description="Get weather information",
    input_schema=WeatherQuery.model_json_schema()
)

# Works with Anthropic
LLM(model_name="claude-sonnet-4", provider="anthropic", tools=Tools(tools=[tool]))

# Works with OpenAI
LLM(model_name="gpt-4o", provider="openai", tools=Tools(tools=[tool]))

# Works with Google
LLM(model_name="gemini-1.5-pro", provider="google", tools=Tools(tools=[tool]))
```

## Best Practices

### Provider Selection Strategy

1. **Primary Provider**: Choose based on your main use case
2. **Backup Providers**: Configure 2-3 alternatives for reliability
3. **Cost Optimization**: Use cheaper models for backup when appropriate
4. **Geographic Considerations**: Choose providers with regional availability

### Multi-Provider Configuration

```python
def get_production_config():
    return LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",  # Primary: Best quality
        backup_models=[
            BackupModel(
                model=LLM(model_name="gpt-4o", provider="openai"),
                priority=1  # Backup: Good balance
            ),
            BackupModel(
                model=LLM(model_name="gemini-1.5-pro", provider="google"),
                priority=2  # Fallback: Cost-effective
            )
        ],
        try_other_providers=True  # Try same model on other providers
    )
```

### Monitoring and Observability

```python
# Track which providers are being used
response = await client.messages.create(messages=messages)
print(f"Used: {response.model} from {response.provider}")

# Monitor token usage across providers
usage_by_provider = {}
usage_by_provider[response.provider] = usage_by_provider.get(response.provider, 0) + response.usage.total_tokens
```

## Provider Roadmap

### Planned Additions
- **AWS Bedrock**: Access to Claude and other models via AWS
- **Ollama**: Local model support for on-premises deployment
- **Cohere**: Additional text generation capabilities
- **Replicate**: Access to open-source models

### Feature Roadmap
- **Streaming**: Real-time response streaming across all providers
- **Batch Processing**: Efficient batch requests
- **Caching**: Response caching for cost optimization
- **Rate Limiting**: Built-in rate limiting and retry logic

## Migration Guide

### From Direct Provider APIs

If you're currently using provider APIs directly:

=== "From Anthropic SDK"
    ```python
    # Before (Anthropic SDK)
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    # After (v-router)
    from v_router import Client, LLM
    client = Client(LLM(model_name="claude-sonnet-4", provider="anthropic"))
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
    ```

=== "From OpenAI SDK"
    ```python
    # Before (OpenAI SDK)
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    # After (v-router)
    from v_router import Client, LLM
    client = Client(LLM(model_name="gpt-4o", provider="openai"))
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
    ```

### Benefits of Migration

1. **Unified Interface**: Same code works with all providers
2. **Automatic Fallback**: Reliability through redundancy
3. **Easy Switching**: Change providers without code changes
4. **Future-Proof**: New providers added without breaking changes

## Next Steps

- [Configure specific providers](anthropic.md)
- [Learn about automatic fallback](../guide/automatic-fallback.md)
- [See provider examples](../examples/provider-comparisons.md)

[Provider Configuration →](../guide/provider-configuration.md){ .md-button .md-button--primary }
[Examples →](../examples/provider-comparisons.md){ .md-button }