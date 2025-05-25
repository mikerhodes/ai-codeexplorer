# Contributing to `codeexplorer`

## Quick Start

`codeexplorer` is an AI tool that enables LLM models to explore codebases using function tools rather than large context windows.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Development Setup

If you intend to contribute code, you'll need to fork the project code, but for simplicity we reference this repository below. Make sure to correct it!

```bash
# Clone and setup
git clone https://github.com/mikerhodes/ai-codeexplorer.git
cd ai-codeexplorer

# Install dependencies and run
uv run codeexplorer --help
```

No separate virtual environment setup needed - `uv run` handles this automatically.

### API Keys

Set environment variables for your preferred AI provider:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-key"

# IBM WatsonX
export WATSONX_IAM_API_KEY="your-key"
export WATSONX_PROJECT="your-project"
export WATSONX_URL="your-url"

# Ollama (local server on default port)
# No keys needed - just run: ollama serve
```

### Testing Your Changes

`codeexplorer` has an entrypoint set in `pyproject.toml` so you can
start it without calling the main code file.

```bash
# Basic functionality test
uv run codeexplorer -p ollama -t "List the main files" .

# Test with edits (requires clean git working directory)
uv run codeexplorer -p ollama --allow-edits -t "Create a simple test file" .
```

### Getting changes merged

Submit a pull request to this repo.

## Architecture

### Core Components

- **`codeexplorer.py`** - Single main module (~700 lines)
- **Provider Adapters** - `OllamaAdapter`, `AnthropicAdapter`, `WatsonxAdapter`
- **Tools** - File operations with path validation and git safety checks

### Key Design Patterns

- **Adapter Pattern** - Unified interface across different AI providers
- **Tool-based Agent** - LLM uses function calls rather than large context
- **Safety First** - Read-only by default, explicit opt-in for edits

### Tool Functions (at time of writing)

| Tool | Purpose |
|------|---------|
| `list_directory_simple` | Recursively list all files in directory tree |
| `read_file_path` | Read file contents with path validation |
| `think` | Log reasoning without taking action |
| `str_replace` | Edit files by exact string replacement |
| `create` | Create new files |

## Code Quality

### pre-commit

This project uses [pre-commit](https://pre-commit.com/).

1. If needed, `pre-commit` can be installed with:

    ```
    uv tool install pre-commit
    ```

1. The repository needs to be initialised to install `pre-commit` git hooks:

    ```
    # in the repo
    pre-commit install
    ```

### Linting & Formatting

```bash
# Check with ruff (configured in pyproject.toml)
uv run ruff check src/

# Type checking with pyright
uv run pyright src/
```

### Style Guidelines

- Line length: 77 characters (ruff configured)
- Python 3.12+ type hints
- Minimal dependencies - avoid adding new ones unless essential

## Project Structure

```
ai-codeexplorer/
├── src/codeexplorer/
│   └── codeexplorer.py          # Main module
├── pyproject.toml               # Package config + tool settings
├── uv.lock                      # Locked dependencies
└── README.md                    # User documentation
```

## Common Tasks

### Adding a New AI Provider

1. Create new adapter class following existing pattern
2. Implement required methods: `prepare_messages`, `chat`, `tools_for_model`, etc.
3. Add provider to `--provider` choices in argument parser
4. Update README with setup instructions

### Adding a New Tool

1. Add function definition to `tools` or `edit_tools` list
2. Implement handler in `process_tool_call()`
3. Ensure proper path validation and safety checks

### Debugging

```bash
# Enable debug logging
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from codeexplorer.codeexplorer import main
main()
" -p ollama .
```

## Release Process

This project uses semantic versioning. Update version in `pyproject.toml` before releasing.

## Questions?

- Check existing issues on GitHub
- Review the main module - it's designed to be readable
- Most logic is in a single file for simplicity
