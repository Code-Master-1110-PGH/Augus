<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents

- [Contributing to AUGUS](#contributing-to-augus)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [Development Setup](#development-setup)
  - [Workflow](#workflow)
  - [Coding Standards](#coding-standards)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Pull Request Process](#pull-request-process)
  - [Commit Guidelines](#commit-guidelines)
  - [Issue Guidelines](#issue-guidelines)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Contributing to AUGUS

First off, thank you for considering contributing to AUGUS! It's people like you that make AUGUS such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

- **New to AUGUS?** Start by reading our [README.md](README.md) to understand the project structure
- **Want to contribute?** Look for [good first issues](https://github.com/Code-Master-1110-PGH/Augus/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [help wanted](https://github.com/Code-Master-1110-PGH/Augus/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- **Questions?** Join our [discussions](https://github.com/Code-Master-1110-PGH/Augus/discussions)

## Development Setup

### Prerequisites

- Python 3.13+
- Git
- Virtual environment tool (venv or conda)

### Local Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Augus.git
cd Augus

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit regularly:
   ```bash
   git commit -m "Clear, descriptive commit message"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:

```bash
# Format code with Black
black app/ tests/

# Check style with Pylint
pylint app/

# Type checking with mypy
mypy app/

# Security check with bandit
bandit -r app/
```

### Code Quality Requirements

- **Minimum Python version**: 3.13+
- **Type hints**: All functions must have type hints
- **Docstrings**: All public functions and classes must have docstrings
- **Line length**: Maximum 100 characters
- **Imports**: Organize using isort

### Documentation

```python
def analyze_financial_data(ticker: str, metrics: list[str]) -> dict[str, float]:
    """
    Analyze financial data for a given company ticker.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        metrics: List of financial metrics to analyze
        
    Returns:
        Dictionary mapping metric names to their analyzed values
        
    Raises:
        ValueError: If ticker is invalid
        KeyError: If metrics are not available
        
    Example:
        >>> analyze_financial_data('AAPL', ['revenue', 'net_income'])
        {'revenue': 365.817e9, 'net_income': 93.736e9}
    """
    # Implementation
    pass
```

## Testing

### Writing Tests

```python
# tests/test_your_feature.py
import pytest
from app.your_module import your_function

class TestYourFeature:
    """Test suite for your feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = your_function()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        result = await your_async_function()
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_graph.py -v

# Run tests matching a pattern
pytest -k "test_financial" -v

# Run with parallel execution
pytest -n auto
```

### Test Coverage Requirements

- Minimum coverage: 80%
- New features must have tests
- Bug fixes must include regression tests

## Documentation

### Writing Docs

- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes
- Add docstrings to all public functions
- Update API documentation in comments

### Building Docs Locally

```bash
# If docs exist, build them
cd docs
pip install -r requirements.txt
make html
```

## Pull Request Process

1. **Update documentation** with your changes
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure CI passes** (all tests, linting, type checks)
5. **Request review** from maintainers
6. **Address feedback** and push updates

### PR Title Format

```
[Type] Brief description

# Examples
[Feature] Add WebSocket support for real-time updates
[Fix] Resolve agent timeout issue in compliance check
[Docs] Update API documentation
[Test] Add integration tests for vector store
```

### PR Description Template

```markdown
## Description
Brief description of your changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality change)
- [ ] Documentation update

## Related Issues
Closes #123

## Testing
Describe how you tested your changes

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented complex areas
- [ ] I have updated the documentation
- [ ] I have added tests that prove my fix is effective
- [ ] New and existing unit tests pass locally
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, missing semicolons)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to build process or dependencies

### Examples

```bash
git commit -m "feat(agents): add sentiment analysis to analyst agent"
git commit -m "fix(api): resolve timeout in research endpoint"
git commit -m "docs: update installation instructions"
git commit -m "test: add integration tests for vector store"
```

## Issue Guidelines

### Reporting Bugs

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Expected behavior
4. Actual behavior

## Environment
- Python version: 3.13
- OS: Windows/macOS/Linux
- AUGUS version: 1.0.0

## Additional Context
Any other context about the problem
```

### Feature Requests

```markdown
## Description
Clear description of the feature

## Motivation
Why this feature would be useful

## Proposed Solution
Suggested implementation approach

## Alternatives
Other approaches you've considered
```

## Performance Considerations

When submitting code:

- Avoid unnecessary imports
- Use generators for large datasets
- Profile slow code paths
- Document performance implications
- Include benchmarks for critical paths

## Security

- Never commit secrets, API keys, or credentials
- Always use environment variables for sensitive data
- Validate and sanitize user inputs
- Use parameterized queries for database operations
- Report security issues privately to maintainers

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors page
- Project documentation

## Questions?

- 📧 Email: support@augus.dev
- 💬 Discussions: [GitHub Discussions](https://github.com/Code-Master-1110-PGH/Augus/discussions)
- 📖 Docs: [Documentation](https://docs.augus.dev)

---

**Thank you for contributing to AUGUS! 🙏**
