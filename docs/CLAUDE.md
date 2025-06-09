# Documentation Setup - CLAUDE.md

This file provides guidance for Claude Code when working with the documentation in this repository.

## Documentation Framework

This project uses **Material for MkDocs** for documentation generation and deployment.

### Framework Details
- **Tool**: Material for MkDocs (https://squidfunk.github.io/mkdocs-material/)
- **Configuration**: `mkdocs.yml` in the root directory
- **Source**: Documentation files in `docs/` directory
- **Output**: Static site generated in `site/` directory

### Local Development
```bash
# Serve documentation locally (with live reload)
uv run mkdocs serve

# Build documentation for production
uv run mkdocs build
```

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages using GitHub Actions.

### Deployment Setup
- **Source**: GitHub Actions (configured in repository settings)
- **Workflow**: `.github/workflows/docs.yml`
- **URL**: https://vectrix-ai.github.io/v-router

### Important Configuration Notes
1. **GitHub Pages Source**: Must be set to "GitHub Actions" in repository settings (not "Deploy from branch")
2. **Jekyll Prevention**: The workflow includes `touch site/.nojekyll` to prevent GitHub from processing the site with Jekyll
3. **Dependencies**: mkdocs-material is installed via `uv sync --all-extras` from pyproject.toml dev dependencies

### Deployment Triggers
- Push to `main` branch with changes to:
  - `docs/**` (any documentation files)
  - `mkdocs.yml` (configuration changes)
  - `.github/workflows/docs.yml` (workflow changes)
- Manual trigger via GitHub Actions UI

### Troubleshooting
If CSS/styling is not working on GitHub Pages:
1. Verify GitHub Pages source is set to "GitHub Actions"
2. Check that `.nojekyll` file is created in the deployment workflow
3. Ensure mkdocs-material is properly installed in the workflow
4. Verify the workflow completes successfully without errors

## File Structure
```
docs/
├── index.md                    # Homepage
├── getting-started/
│   ├── installation.md
│   ├── quick-start.md
│   └── configuration.md
├── guide/
│   └── function-calling.md
├── api/
│   ├── client.md
│   └── llm.md
├── providers/
│   └── overview.md
├── examples/
│   └── basic.md
└── stylesheets/
    └── extra.css              # Custom CSS overrides
```

## Development Guidelines
- Use Material for MkDocs components and extensions as configured in mkdocs.yml
- Follow the existing navigation structure defined in mkdocs.yml
- Place custom CSS in `docs/stylesheets/extra.css`
- Use proper markdown syntax with Material extensions (admonitions, code blocks, etc.)