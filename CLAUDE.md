# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/models/test_llm.py

# Run tests with verbose output
uv run pytest -v
```

### Linting and Formatting
```bash
# Run linter (checks code style issues)
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Package Management
```bash
# Install dependencies (including dev dependencies)
uv sync --all-extras

# Add a new dependency
uv run uv add <package-name>

# Add a dev dependency
uv run uv add --dev <package-name>
```

## Python Code Execution
- When trying to run Python code always use UV (uv run example.py)

## Architecture Overview

v-router is a unified LLM interface that provides automatic fallback between different LLM providers. The architecture follows a provider pattern with these key components:

### Core Components

1. **Client** (`src/v_router/client.py`): Main entry point that exposes a messages API similar to provider SDKs.

2. **Router** (`src/v_router/router.py`): Handles request routing, fallback logic, and provider selection. Key features:
   - Maintains provider instances
   - Handles model name mappings via `models.yml`
   - Implements fallback strategies (backup models, cross-provider fallback)

3. **Providers** (`src/v_router/providers/`): Each provider inherits from `BaseProvider` and implements:
   - `create_message()`: Sends requests to the provider's API
   - `name`: Provider identifier
   - Model name validation/mapping

4. **Model Configuration** (`src/v_router/models.yml`): YAML file that maps model names to:
   - Available providers
   - Provider-specific model name mappings (e.g., `claude-3-opus` → `claude-3-opus@20240229` for Vertex)

### Request Flow

1. User creates a `Client` with an `LLM` configuration
2. Client receives messages and passes to `Router`
3. Router attempts primary model/provider
4. On failure, Router tries backup models in priority order
5. If `try_other_providers=True`, Router attempts same model on alternative providers
6. Response returned in unified `Response` format

### Provider Integration

When adding a new provider:
1. Create provider class in `src/v_router/providers/`
2. Inherit from `BaseProvider`
3. Implement `create_message()` and `name` property
4. Add to `PROVIDER_REGISTRY` in `router.py`
5. Update `models.yml` with supported models

## Development Guidelines

- Don't use relative imports unless absolutely needed, try to use absolute ones

## Memories
- Remember that we use mkdocs in a docs directory if I ask to write documentation
- We use Material for MkDocs, it should be built using GitHub Actions with this workflow:
```yaml
name: ci 
on:
  push:
    branches:
      - master 
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure Git Credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV 
      - uses: actions/cache@v4
        with:
          key: mkdocs-material-${{ env.cache_id }}
          path: .cache 
          restore-keys: |
            mkdocs-material-
      - run: pip install mkdocs-material 
      - run: mkdocs gh-deploy --force
```