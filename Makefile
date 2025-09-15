.PHONY: setup lint test clean help

# Default target executed when no arguments are given to make.
all: help

# Define variables
PYTHON := python3
PIP := pip3
JUPYTER := jupyter
LINTERS := black flake8 isort

help:
	@echo "Available commands:"
	@echo "setup      : Install required packages"
	@echo "lint       : Run code linters (black, flake8, isort)"
	@echo "test       : Run tests"
	@echo "notebook   : Run Jupyter notebook"
	@echo "clean      : Remove Python and build artifacts"
	@echo "clean-all  : Remove all artifacts including dependencies"

# Setup development environment
setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -U $(LINTERS) pytest nbval jupyter

# Lint the code
lint:
	black --check .
	flake8 .
	isort --check-only --profile black .

# Run tests (including notebook tests)
test:
	pytest --nbval Interpretable_ML_Chris.ipynb
	pytest

# Run Jupyter notebook
notebook:
	$(JUPYTER) notebook

# Clean Python artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".benchmarks" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .ipynb_checkpoints

# Clean all artifacts including dependencies
clean-all: clean
	rm -rf venv/
	rm -rf .venv/