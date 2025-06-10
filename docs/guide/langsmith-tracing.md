# LangSmith Tracing

v-router supports [LangSmith](https://smith.langchain.com/) tracing for multiple providers, allowing you to monitor and debug your LLM applications with detailed observability.

## Installation

To use LangSmith tracing, install v-router with the `tracing` extra:

```bash
pip install 'v-router[tracing]'
```

## Quick Start

### 1. Set up environment variables

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"

# Optional: Set a project name
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

### 2. Use v-router as normal

Once tracing is enabled, all LLM API calls will be automatically traced:

```python
from v_router import Client, LLM

client = Client(
    llm_config=LLM(
        model_name="gpt-4o",
        provider="openai",
        max_tokens=1000
    )
)

# This call will be traced in LangSmith
response = await client.messages.create(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## How It Works

When `LANGCHAIN_TRACING_V2` is set to `"true"`, v-router automatically wraps supported clients with LangSmith's tracing wrapper. This happens transparently during provider initialization:

- If LangSmith is installed and tracing is enabled, the client is wrapped
- If LangSmith is not installed, v-router continues to work normally without tracing
- If tracing is disabled, the client is not wrapped

## Supported Providers

Currently, LangSmith tracing is supported for:

- `anthropic` - Direct Anthropic API
- `vertexai` - Anthropic via Google Vertex AI
- `openai` - Direct OpenAI API
- `azure` - OpenAI via Azure OpenAI

## Advanced Usage

### Using with @traceable decorator

You can combine v-router with LangSmith's `@traceable` decorator for custom function tracing:

```python
from langsmith import traceable

@traceable(name="Question Answering")
async def answer_question(question: str) -> str:
    response = await client.messages.create(
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question}
        ]
    )
    return response.content[0].text

# Both the function and the LLM call will be traced
answer = await answer_question("What is the capital of France?")
```

### Tracing with Tool Use

Tool calls are also traced, providing visibility into the entire conversation flow:

```python
from v_router.classes.tools import Tool, Tools

# Define a tool
weather_tool = Tool(
    name="get_weather",
    description="Get current weather",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
)

# Use with tracing
response = await client.messages.create(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=Tools(tools=[weather_tool]),
    tool_choice="auto"
)
```

## Viewing Traces

After running your application with tracing enabled:

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to your project
3. View detailed traces including:
   - Input/output data
   - Latency metrics  
   - Token usage
   - Error tracking
   - Nested call structure

## Configuration Options

### Environment Variables

- `LANGCHAIN_TRACING_V2`: Set to `"true"` to enable tracing
- `LANGCHAIN_API_KEY`: Your LangSmith API key
- `LANGCHAIN_PROJECT`: Project name for organizing traces (optional)
- `LANGCHAIN_ENDPOINT`: Custom LangSmith endpoint (optional)

### Disabling Tracing

To disable tracing:

```python
# Set to false
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Or remove the variable
del os.environ["LANGCHAIN_TRACING_V2"]
```

## Best Practices

1. **Development vs Production**: Consider enabling tracing only in development/staging environments
2. **Sensitive Data**: Be aware that traced data is sent to LangSmith - ensure you're not logging sensitive information
3. **Performance**: Tracing adds minimal overhead, but consider disabling in high-throughput production scenarios
4. **Project Organization**: Use different project names for different environments or applications

## Troubleshooting

### Tracing not working

1. Verify LangSmith is installed: `pip install langsmith`
2. Check environment variables are set correctly
3. Ensure you have a valid LangSmith API key
4. Verify your LangSmith account is active

### Import errors

If you see import errors, install the tracing extra:

```bash
pip install 'v-router[tracing]'
```

## Example Notebook

For a complete example, see the [LangSmith tracing notebook](https://github.com/your-repo/v-router/blob/main/examples/langsmith_tracing.ipynb).