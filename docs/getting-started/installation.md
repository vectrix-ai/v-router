# Installation

v-router is available on PyPI and can be installed using pip or other Python package managers.

## Prerequisites

- **Python 3.13.3+**: v-router requires Python 3.13.3 or higher
- **API Keys**: You'll need API keys for the providers you want to use

## Install with pip

```bash
pip install v-router
```

## Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
uv add v-router
```

## Development Installation

If you want to contribute to v-router or run the latest development version:

```bash
# Clone the repository
git clone https://github.com/vectrix-ai/v-router.git
cd v-router

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## Verify Installation

Test your installation by running:

```python
import v_router
print(v_router.__version__)
```

Or check the available components:

```python
from v_router import Client, LLM, BackupModel
print("v-router installed successfully!")
```

## Provider Dependencies

v-router automatically installs the required dependencies for all supported providers:

- **Anthropic**: `anthropic[vertex]>=0.52.0`
- **OpenAI**: `openai>=1.82.0`
- **Google AI**: `google-genai>=1.16.1`
- **Google Cloud (Vertex AI)**: `google-cloud-aiplatform>=1.94.0`
- **Configuration**: `pyyaml>=6.0.2`
- **Logging**: `colorlog>=6.9.0`

All dependencies are installed automatically when you install v-router.

## Optional Dependencies

For development and testing:

```bash
# Development tools
uv add --dev pytest pytest-asyncio ruff pre-commit

# Documentation
uv add --dev mkdocs-material

# Jupyter notebooks (for examples)
uv add --dev ipykernel python-dotenv
```

## System Requirements

- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimum 512MB RAM
- **Network**: Internet connection required for API calls

## Troubleshooting

### Common Issues

#### ImportError: No module named 'v_router'

Make sure you've installed v-router in the correct Python environment:

```bash
# Check your Python version
python --version

# Check installed packages
pip list | grep v-router
```

#### ModuleNotFoundError for provider libraries

If you see errors about missing provider libraries, try reinstalling:

```bash
pip install --upgrade v-router
```

#### API Key Issues

Make sure your environment variables are set correctly. See the [Configuration](configuration.md) guide for details.

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](../development/contributing.md#troubleshooting)
2. Search [existing issues](https://github.com/vectrix-ai/v-router/issues)
3. Create a [new issue](https://github.com/vectrix-ai/v-router/issues/new) with:
   - Python version
   - v-router version
   - Operating system
   - Full error message
   - Minimal code example

## Next Steps

Now that you have v-router installed, you can:

1. [Set up your configuration](configuration.md)
2. [Try the quick start guide](quick-start.md)
3. [Explore the examples](../examples/basic.md)

[Continue to Configuration →](configuration.md){ .md-button .md-button--primary }