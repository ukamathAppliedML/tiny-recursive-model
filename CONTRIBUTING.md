# Contributing to Tiny Recursive Model

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/tiny-recursive-model.git
cd tiny-recursive-model
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tiny_recursive_model --cov-report=html

# Run specific test file
pytest tests/test_trm.py -v
```

### Code Formatting

```bash
# Format code
black tiny_recursive_model/ tests/

# Sort imports
ruff check --fix tiny_recursive_model/ tests/

# Type checking
mypy tiny_recursive_model/
```

### Linting

```bash
ruff check tiny_recursive_model/ tests/
```

## Adding a New Task

1. Create a new file in `tiny_recursive_model/data/` or `examples/`
2. Extend `BaseDataset` with your task's data generation
3. Register the task using `@register_task("your_task")`
4. Add tests in `tests/`
5. Update documentation

See `examples/custom_task_template.py` for a complete template.

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Linting

```bash
pytest
ruff check .
black --check .
mypy tiny_recursive_model/
```

### 4. Commit with Clear Messages

```bash
git commit -m "feat: add new maze task dataset"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guide

### Python

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Maximum line length: 100 characters

### Documentation

- Use clear, concise language
- Include examples in docstrings
- Update README for significant changes

### Examples

```python
def my_function(
    param1: int,
    param2: str = "default"
) -> torch.Tensor:
    """Brief description of function.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2
    
    Returns
    -------
    torch.Tensor
        Description of return value
    
    Examples
    --------
    >>> result = my_function(42, "hello")
    >>> print(result.shape)
    torch.Size([10])
    """
    pass
```

## Reporting Issues

### Bug Reports

Please include:
- Python version
- PyTorch version
- Operating system
- Minimal reproducible example
- Full error traceback

### Feature Requests

Please include:
- Clear description of the feature
- Use case / motivation
- (Optional) Proposed implementation

## Questions?

- Open a GitHub Issue for questions
- Tag with `question` label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
