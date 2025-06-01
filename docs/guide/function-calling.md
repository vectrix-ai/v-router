# Function Calling

v-router provides unified function calling (tool use) across all providers. Define your tools once and use them with any LLM, whether it's Anthropic's Claude, OpenAI's GPT models, or Google's Gemini.

## Quick Start

Here's a simple example that works across all providers:

```python
import asyncio
from pydantic import BaseModel, Field
from v_router import Client, LLM
from v_router.classes.tools import ToolCall, Tools

# Define tool schema using Pydantic
class WeatherQuery(BaseModel):
    location: str = Field(..., description="City and state, e.g. San Francisco, CA")
    units: str = Field("fahrenheit", description="Temperature units (celsius/fahrenheit)")

# Create tool definition
weather_tool = ToolCall(
    name="get_weather",
    description="Get current weather for a location",
    input_schema=WeatherQuery.model_json_schema()
)

async def main():
    # Configure LLM with tools
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        tools=Tools(tools=[weather_tool])
    )
    
    client = Client(llm_config)
    
    # Make request that will trigger tool use
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": "What's the weather like in Paris, France?"}
        ]
    )
    
    # Check for tool calls
    if response.tool_use:
        for tool_call in response.tool_use:
            print(f"Tool: {tool_call.name}")
            print(f"Arguments: {tool_call.arguments}")
            
            # In a real app, you'd execute the function here
            if tool_call.name == "get_weather":
                # Simulate weather API call
                weather_data = {"temperature": "22°C", "condition": "sunny"}
                print(f"Weather result: {weather_data}")
    else:
        print(f"Response: {response.content[0].text}")

asyncio.run(main())
```

## Tool Definition

### Using Pydantic (Recommended)

Pydantic provides type safety and automatic schema generation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from v_router.classes.tools import ToolCall

class DatabaseQuery(BaseModel):
    table: str = Field(..., description="Database table name")
    columns: List[str] = Field(..., description="Columns to select")
    where_clause: Optional[str] = Field(None, description="SQL WHERE clause")
    limit: int = Field(10, description="Maximum number of results")

db_tool = ToolCall(
    name="query_database",
    description="Execute a SELECT query on the database",
    input_schema=DatabaseQuery.model_json_schema()
)
```

### Manual Schema Definition

You can also define schemas manually:

```python
from v_router.classes.tools import ToolCall

manual_tool = ToolCall(
    name="calculate_distance",
    description="Calculate distance between two points",
    input_schema={
        "type": "object",
        "properties": {
            "lat1": {"type": "number", "description": "Latitude of first point"},
            "lon1": {"type": "number", "description": "Longitude of first point"},
            "lat2": {"type": "number", "description": "Latitude of second point"},
            "lon2": {"type": "number", "description": "Longitude of second point"}
        },
        "required": ["lat1", "lon1", "lat2", "lon2"]
    }
)
```

## Multiple Tools

Configure multiple tools for complex workflows:

```python
from pydantic import BaseModel, Field
from v_router.classes.tools import ToolCall, Tools

# File operations
class FileOperation(BaseModel):
    path: str = Field(..., description="File path")
    content: Optional[str] = Field(None, description="File content for write operations")

# Email operations  
class EmailMessage(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")

# Mathematical operations
class MathOperation(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

# Create tools
file_tool = ToolCall(
    name="file_operations",
    description="Read or write files",
    input_schema=FileOperation.model_json_schema()
)

email_tool = ToolCall(
    name="send_email", 
    description="Send an email message",
    input_schema=EmailMessage.model_json_schema()
)

math_tool = ToolCall(
    name="calculate",
    description="Perform mathematical calculations",
    input_schema=MathOperation.model_json_schema()
)

# Configure LLM with multiple tools
llm_config = LLM(
    model_name="gpt-4o",
    provider="openai",
    tools=Tools(tools=[file_tool, email_tool, math_tool])
)
```

## Tool Execution Loop

Here's a complete example showing how to handle tool calls and continue the conversation:

```python
import asyncio
import json
from v_router import Client, LLM
from v_router.classes.tools import ToolCall, Tools
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

class WebSearch(BaseModel):
    query: str = Field(..., description="Search query")

async def execute_tool(tool_name: str, arguments: dict):
    """Execute the requested tool and return results."""
    if tool_name == "calculate":
        try:
            # Safe evaluation of simple math expressions
            result = eval(arguments["expression"])
            return {"result": result}
        except Exception as e:
            return {"error": f"Calculation error: {e}"}
    
    elif tool_name == "web_search":
        # Simulate web search
        query = arguments["query"]
        return {
            "results": [
                {"title": f"Result for {query}", "url": "https://example.com"},
                {"title": f"More about {query}", "url": "https://example.org"}
            ]
        }
    
    return {"error": f"Unknown tool: {tool_name}"}

async def chat_with_tools():
    # Define tools
    calc_tool = ToolCall(
        name="calculate",
        description="Perform mathematical calculations",
        input_schema=Calculator.model_json_schema()
    )
    
    search_tool = ToolCall(
        name="web_search", 
        description="Search the web for information",
        input_schema=WebSearch.model_json_schema()
    )
    
    # Create client with tools
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        tools=Tools(tools=[calc_tool, search_tool])
    )
    
    client = Client(llm_config)
    
    # Start conversation
    messages = [
        {"role": "user", "content": "What is 15 * 23, and can you search for information about quantum computing?"}
    ]
    
    while True:
        response = await client.messages.create(messages=messages)
        
        # Add assistant message
        messages.append({
            "role": "assistant", 
            "content": response.content[0].text if response.content else ""
        })
        
        # Handle tool calls
        if response.tool_use:
            for tool_call in response.tool_use:
                print(f"🔧 Executing {tool_call.name}...")
                
                # Execute the tool
                result = await execute_tool(tool_call.name, tool_call.arguments)
                
                # Add tool result to conversation
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_call.name}: {json.dumps(result)}"
                })
            
            # Continue conversation with tool results
            continue
        else:
            # No more tool calls, show final response
            print(f"🤖 {response.content[0].text}")
            break

asyncio.run(chat_with_tools())
```

## Provider Compatibility

All function calling features work across providers:

=== "Anthropic Claude"

    ```python
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        tools=Tools(tools=[your_tool])
    )
    ```

=== "OpenAI GPT"

    ```python
    llm_config = LLM(
        model_name="gpt-4o",
        provider="openai", 
        tools=Tools(tools=[your_tool])
    )
    ```

=== "Google Gemini"

    ```python
    llm_config = LLM(
        model_name="gemini-1.5-pro",
        provider="google",
        tools=Tools(tools=[your_tool])
    )
    ```

=== "Vertex AI"

    ```python
    # Works with both Claude and Gemini on Vertex
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="vertexai",
        tools=Tools(tools=[your_tool])
    )
    ```

## Advanced Tool Patterns

### Conditional Tool Availability

Control which tools are available based on context:

```python
def get_tools_for_user(user_role: str) -> Tools:
    """Return tools based on user permissions."""
    base_tools = [search_tool, weather_tool]
    
    if user_role == "admin":
        base_tools.extend([file_tool, database_tool])
    elif user_role == "developer":
        base_tools.extend([code_tool, git_tool])
    
    return Tools(tools=base_tools)

# Use conditional tools
user_tools = get_tools_for_user("admin")
llm_config = LLM(
    model_name="claude-sonnet-4",
    provider="anthropic",
    tools=user_tools
)
```

### Tool Chaining

Chain tools together for complex workflows:

```python
async def research_and_summarize(topic: str):
    """Chain web search and summarization tools."""
    
    # Step 1: Search for information
    search_response = await client.messages.create(
        messages=[{
            "role": "user", 
            "content": f"Search for recent information about {topic}"
        }]
    )
    
    # Execute search tool
    search_results = []
    if search_response.tool_use:
        for tool_call in search_response.tool_use:
            if tool_call.name == "web_search":
                results = await execute_tool(tool_call.name, tool_call.arguments)
                search_results.extend(results.get("results", []))
    
    # Step 2: Summarize findings
    summary_response = await client.messages.create(
        messages=[{
            "role": "user",
            "content": f"Summarize these search results about {topic}: {search_results}"
        }]
    )
    
    return summary_response.content[0].text
```

### Error Handling in Tools

Implement robust error handling:

```python
async def safe_tool_execution(tool_name: str, arguments: dict):
    """Execute tools with comprehensive error handling."""
    try:
        if tool_name == "risky_operation":
            # Validate inputs
            if not arguments.get("required_param"):
                return {"error": "Missing required parameter"}
            
            # Perform operation with timeout
            import asyncio
            result = await asyncio.wait_for(
                risky_operation(arguments),
                timeout=30.0
            )
            
            return {"success": True, "result": result}
            
    except asyncio.TimeoutError:
        return {"error": "Operation timed out"}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}
```

## Real-World Examples

### Code Analysis Tool

```python
from pydantic import BaseModel, Field

class CodeAnalysis(BaseModel):
    code: str = Field(..., description="Code to analyze")
    language: str = Field("python", description="Programming language")
    analysis_type: str = Field("quality", description="Type: quality, security, performance")

code_tool = ToolCall(
    name="analyze_code",
    description="Analyze code for quality, security, or performance issues",
    input_schema=CodeAnalysis.model_json_schema()
)

async def analyze_code(code: str, language: str, analysis_type: str):
    """Implement actual code analysis logic."""
    # This would integrate with tools like pylint, bandit, etc.
    return {
        "issues": ["Line 5: Unused variable", "Line 12: Potential security risk"],
        "score": 85,
        "recommendations": ["Use more descriptive variable names"]
    }
```

### Database Operations

```python
class DatabaseQuery(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    database: str = Field("main", description="Database name")
    read_only: bool = Field(True, description="Whether query is read-only")

db_tool = ToolCall(
    name="execute_sql",
    description="Execute SQL queries on the database",
    input_schema=DatabaseQuery.model_json_schema()
)

async def execute_sql(query: str, database: str, read_only: bool):
    """Execute SQL with safety checks."""
    if not read_only and "SELECT" not in query.upper():
        return {"error": "Write operations require read_only=False"}
    
    # Execute query safely...
    return {"rows": [], "affected": 0}
```

## Best Practices

### Tool Design

1. **Clear Descriptions**: Make tool purposes obvious
2. **Strong Typing**: Use Pydantic for schema validation
3. **Error Handling**: Return structured error responses
4. **Documentation**: Include examples in descriptions

### Performance

1. **Async Operations**: Use async/await for I/O operations
2. **Timeouts**: Set reasonable timeouts for external calls
3. **Caching**: Cache frequently accessed data
4. **Parallel Execution**: Execute independent tools concurrently

### Security

1. **Input Validation**: Validate all tool inputs
2. **Permissions**: Check user permissions before tool execution
3. **Sandboxing**: Isolate dangerous operations
4. **Audit Logging**: Log all tool executions

## Troubleshooting

### Common Issues

#### Tool Not Called

```python
# ❌ Vague description
ToolCall(name="helper", description="Helps with stuff")

# ✅ Clear description  
ToolCall(name="calculate_tax", description="Calculate tax amount for a given income and tax rate")
```

#### Schema Errors

```python
# ❌ Missing required fields
{"type": "object", "properties": {}}

# ✅ Complete schema
{
    "type": "object", 
    "properties": {"amount": {"type": "number"}},
    "required": ["amount"]
}
```

#### Tool Execution Failures

```python
# Always return structured responses
async def execute_tool(name, args):
    try:
        result = await actual_tool_function(args)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Next Steps

- [Explore multimodal content](multimodal-content.md)
- [Learn about provider configuration](provider-configuration.md)
- [See advanced examples](../examples/advanced.md)

[Multimodal Content →](multimodal-content.md){ .md-button .md-button--primary }
[Advanced Examples →](../examples/advanced.md){ .md-button }