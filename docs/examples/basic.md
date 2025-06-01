# Basic Examples

This page provides basic examples to help you get started with v-router. These examples cover the most common use cases and patterns.

## Simple Chat Completion

The most basic use case - sending a message and getting a response:

```python
import asyncio
from v_router import Client, LLM

async def simple_chat():
    # Configure LLM
    llm_config = LLM(
        model_name="claude-sonnet-4",
        provider="anthropic",
        max_tokens=500
    )
    
    # Create client
    client = Client(llm_config)
    
    # Send message
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": "What is machine learning?"}
        ]
    )
    
    print(f"Response: {response.content[0].text}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens}")

# Run the example
asyncio.run(simple_chat())
```

## Multi-Turn Conversation

Building a conversation with multiple exchanges:

```python
import asyncio
from v_router import Client, LLM

async def conversation_example():
    client = Client(
        LLM(model_name="gpt-4o", provider="openai")
    )
    
    # Start conversation
    messages = [
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "How do I create a list in Python?"}
    ]
    
    # First response
    response = await client.messages.create(messages=messages)
    print(f"Assistant: {response.content[0].text}")
    
    # Add assistant response to conversation
    messages.append({
        "role": "assistant", 
        "content": response.content[0].text
    })
    
    # Continue conversation
    messages.append({
        "role": "user", 
        "content": "Can you show me how to add items to it?"
    })
    
    # Second response
    response = await client.messages.create(messages=messages)
    print(f"Assistant: {response.content[0].text}")

asyncio.run(conversation_example())
```

## Provider Comparison

Try the same request with different providers:

```python
import asyncio
from v_router import Client, LLM

async def compare_providers():
    prompt = "Explain quantum computing in exactly 50 words."
    
    providers = [
        ("claude-sonnet-4", "anthropic"),
        ("gpt-4o", "openai"),
        ("gemini-1.5-pro", "google")
    ]
    
    for model_name, provider in providers:
        client = Client(
            LLM(
                model_name=model_name,
                provider=provider,
                max_tokens=100
            )
        )
        
        response = await client.messages.create(
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(f"\n{provider.upper()} ({response.model}):")
        print(response.content[0].text)
        print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(compare_providers())
```

## Temperature and Creativity Control

Comparing different temperature settings:

```python
import asyncio
from v_router import Client, LLM

async def temperature_comparison():
    prompt = "Write a short creative story about a robot."
    
    temperatures = [0.1, 0.5, 0.9]
    
    for temp in temperatures:
        client = Client(
            LLM(
                model_name="claude-sonnet-4",
                provider="anthropic",
                temperature=temp,
                max_tokens=200
            )
        )
        
        response = await client.messages.create(
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(f"\n--- Temperature: {temp} ---")
        print(response.content[0].text)

asyncio.run(temperature_comparison())
```

## Automatic Fallback Example

Demonstrating automatic fallback when the primary model fails:

```python
import asyncio
from v_router import Client, LLM, BackupModel

async def fallback_example():
    # Configure with a model that doesn't exist as primary
    llm_config = LLM(
        model_name="claude-ultra-5",  # This doesn't exist (yet!)
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
    
    try:
        response = await client.messages.create(
            messages=[{"role": "user", "content": "Hello! What model are you?"}]
        )
        
        print(f"Success! Used: {response.model} from {response.provider}")
        print(f"Response: {response.content[0].text}")
        
    except Exception as e:
        print(f"All providers failed: {e}")

asyncio.run(fallback_example())
```

## Cross-Provider Model Testing

Test the same model across different providers:

```python
import asyncio
from v_router import Client, LLM

async def cross_provider_test():
    model_name = "claude-sonnet-4"
    providers = ["anthropic", "vertexai"]  # Same model, different providers
    
    prompt = "Explain the benefits of using multiple LLM providers."
    
    for provider in providers:
        try:
            client = Client(
                LLM(model_name=model_name, provider=provider)
            )
            
            response = await client.messages.create(
                messages=[{"role": "user", "content": prompt}]
            )
            
            print(f"\n{provider.upper()}:")
            print(f"Model: {response.model}")
            print(f"Response: {response.content[0].text[:200]}...")
            
        except Exception as e:
            print(f"{provider} failed: {e}")

asyncio.run(cross_provider_test())
```

## System Prompts and Personas

Using system prompts to create different AI personas:

```python
import asyncio
from v_router import Client, LLM

async def persona_examples():
    personas = {
        "helpful_assistant": "You are a helpful and friendly assistant.",
        "expert_programmer": "You are an expert programmer who explains code clearly and concisely.",
        "creative_writer": "You are a creative writer who tells engaging stories with vivid descriptions.",
        "data_scientist": "You are a data scientist who explains complex concepts in simple terms."
    }
    
    question = "How would you approach solving a complex problem?"
    
    for persona_name, system_prompt in personas.items():
        client = Client(
            LLM(
                model_name="gpt-4o",
                provider="openai",
                max_tokens=150
            )
        )
        
        response = await client.messages.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        
        print(f"\n--- {persona_name.replace('_', ' ').title()} ---")
        print(response.content[0].text)

asyncio.run(persona_examples())
```

## Token Usage Monitoring

Monitor and control token usage across requests:

```python
import asyncio
from v_router import Client, LLM

async def token_monitoring():
    client = Client(
        LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=100  # Limit response length
        )
    )
    
    requests = [
        "What is AI?",
        "Explain machine learning in detail.",
        "Write a haiku about programming.",
        "What are the benefits of cloud computing?"
    ]
    
    total_tokens = 0
    
    for i, prompt in enumerate(requests, 1):
        response = await client.messages.create(
            messages=[{"role": "user", "content": prompt}]
        )
        
        usage = response.usage
        total_tokens += usage.total_tokens
        
        print(f"\nRequest {i}: {prompt}")
        print(f"Response: {response.content[0].text}")
        print(f"Tokens - Input: {usage.input_tokens}, Output: {usage.output_tokens}, Total: {usage.total_tokens}")
    
    print(f"\nTotal tokens used across all requests: {total_tokens}")

asyncio.run(token_monitoring())
```

## Error Handling Patterns

Proper error handling with v-router:

```python
import asyncio
from v_router import Client, LLM

async def error_handling_example():
    # Configuration that might fail
    client = Client(
        LLM(
            model_name="nonexistent-model",
            provider="anthropic"
        )
    )
    
    try:
        response = await client.messages.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Success: {response.content[0].text}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except ConnectionError as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    # Better approach: Use fallback models
    robust_client = Client(
        LLM(
            model_name="nonexistent-model",
            provider="anthropic",
            backup_models=[
                BackupModel(
                    model=LLM(model_name="gpt-4o", provider="openai"),
                    priority=1
                )
            ]
        )
    )
    
    try:
        response = await robust_client.messages.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Fallback success: {response.model} - {response.content[0].text}")
        
    except Exception as e:
        print(f"All providers failed: {e}")

asyncio.run(error_handling_example())
```

## Batch Processing

Process multiple requests efficiently:

```python
import asyncio
from v_router import Client, LLM

async def batch_processing():
    client = Client(
        LLM(model_name="gemini-1.5-flash", provider="google")  # Fast model for batch processing
    )
    
    # Batch of questions
    questions = [
        "What is Python?",
        "What is JavaScript?", 
        "What is Rust?",
        "What is Go?",
        "What is Swift?"
    ]
    
    # Process all questions concurrently
    async def process_question(question):
        response = await client.messages.create(
            messages=[{"role": "user", "content": question}]
        )
        return {
            "question": question,
            "answer": response.content[0].text,
            "tokens": response.usage.total_tokens
        }
    
    # Run all requests concurrently
    results = await asyncio.gather(*[
        process_question(q) for q in questions
    ])
    
    # Print results
    total_tokens = 0
    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Tokens: {result['tokens']}")
        total_tokens += result['tokens']
    
    print(f"\nTotal tokens used: {total_tokens}")

asyncio.run(batch_processing())
```

## Configuration Management

Manage different configurations for different use cases:

```python
import asyncio
from v_router import Client, LLM, BackupModel

class ConfigManager:
    @staticmethod
    def get_development_config():
        """Fast, cheap configuration for development."""
        return LLM(
            model_name="gpt-3.5",
            provider="openai",
            max_tokens=500,
            temperature=0.7
        )
    
    @staticmethod
    def get_production_config():
        """Robust configuration with fallbacks for production."""
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=2000,
            temperature=0.3,
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
            try_other_providers=True
        )
    
    @staticmethod
    def get_creative_config():
        """High creativity configuration for creative tasks."""
        return LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=3000,
            temperature=0.9
        )

async def config_management_example():
    import os
    
    # Choose configuration based on environment
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        config = ConfigManager.get_production_config()
    elif env == "creative":
        config = ConfigManager.get_creative_config()
    else:
        config = ConfigManager.get_development_config()
    
    client = Client(config)
    
    response = await client.messages.create(
        messages=[{"role": "user", "content": "Tell me about v-router"}]
    )
    
    print(f"Environment: {env}")
    print(f"Model used: {response.model}")
    print(f"Response: {response.content[0].text}")

asyncio.run(config_management_example())
```

## Performance Comparison

Compare response times across providers:

```python
import asyncio
import time
from v_router import Client, LLM

async def performance_comparison():
    providers_configs = [
        ("claude-sonnet-4", "anthropic"),
        ("gpt-4o", "openai"),
        ("gemini-1.5-flash", "google")  # Flash is optimized for speed
    ]
    
    prompt = "Explain the concept of recursion in programming."
    
    results = []
    
    for model_name, provider in providers_configs:
        client = Client(
            LLM(
                model_name=model_name,
                provider=provider,
                max_tokens=200
            )
        )
        
        # Measure response time
        start_time = time.time()
        
        try:
            response = await client.messages.create(
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            results.append({
                "provider": provider,
                "model": response.model,
                "response_time": response_time,
                "tokens": response.usage.total_tokens,
                "tokens_per_second": response.usage.total_tokens / response_time
            })
            
        except Exception as e:
            print(f"{provider} failed: {e}")
    
    # Print performance comparison
    print("Performance Comparison:")
    print("-" * 80)
    print(f"{'Provider':<15} {'Model':<25} {'Time (s)':<10} {'Tokens':<8} {'Tok/sec':<8}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['response_time']):
        print(f"{result['provider']:<15} {result['model']:<25} {result['response_time']:<10.2f} {result['tokens']:<8} {result['tokens_per_second']:<8.1f}")

asyncio.run(performance_comparison())
```

## Best Practices Example

A comprehensive example showing best practices:

```python
import asyncio
import logging
import os
from v_router import Client, LLM, BackupModel, setup_logger

# Set up logging
setup_logger("v_router", level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionClient:
    def __init__(self):
        self.client = self._create_client()
    
    def _create_client(self):
        """Create a production-ready client configuration."""
        config = LLM(
            model_name="claude-sonnet-4",
            provider="anthropic",
            max_tokens=2000,
            temperature=0.3,
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
            ],
            try_other_providers=True
        )
        
        return Client(config)
    
    async def process_request(self, messages, max_retries=3):
        """Process a request with retry logic and monitoring."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing request (attempt {attempt + 1})")
                
                response = await self.client.messages.create(messages=messages)
                
                # Log successful request
                logger.info(
                    f"Request successful: {response.model} ({response.provider}), "
                    f"tokens: {response.usage.total_tokens}"
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def best_practices_example():
    # Check environment variables
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Missing API keys: {missing_keys}")
        return
    
    client = ProductionClient()
    
    # Process a request
    try:
        response = await client.process_request(
            messages=[
                {"role": "user", "content": "Explain the benefits of using v-router for production LLM applications."}
            ]
        )
        
        print(f"Success: {response.content[0].text}")
        
    except Exception as e:
        print(f"Request failed after all retries: {e}")

# Run the example
asyncio.run(best_practices_example())
```

## Next Steps

These basic examples should help you get started with v-router. For more advanced use cases, check out:

- [Function Calling Examples](../guide/function-calling.md)
- [Multimodal Content Examples](../guide/multimodal-content.md)
- [Advanced Patterns](advanced.md)

[Function Calling →](../guide/function-calling.md){ .md-button .md-button--primary }
[Advanced Examples →](advanced.md){ .md-button }