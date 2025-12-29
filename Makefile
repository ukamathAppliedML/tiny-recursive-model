.PHONY: help install install-dev test lint format clean train demo

help:
	@echo "Tiny Recursive Model - Available commands:"
	@echo ""
	@echo "  make install      Install package"
	@echo "  make install-dev  Install with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make train        Train on Sudoku (quick demo)"
	@echo "  make demo         Run architecture demo"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check tiny_recursive_model/ tests/
	mypy tiny_recursive_model/

format:
	black tiny_recursive_model/ tests/ examples/
	ruff check --fix tiny_recursive_model/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf outputs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:
	python -m tiny_recursive_model train

demo:
	python -m tiny_recursive_model demo

# Docker (optional)
docker-build:
	docker build -t trm .

docker-run:
	docker run -it --rm trm
