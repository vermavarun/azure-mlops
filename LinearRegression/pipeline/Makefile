.PHONY: help setup install install-dev local azure demo clean format lint test

help:
	@echo "MLOps Pipeline - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "make setup        - Initialize environment"
	@echo "make install      - Install dependencies"
	@echo "make install-dev  - Install dev dependencies"
	@echo "make local        - Run pipeline locally"
	@echo "make azure        - Run pipeline on Azure ML"
	@echo "make demo         - Run demo script"
	@echo "make clean        - Clean output files"
	@echo "make format       - Format code with Black"
	@echo "make lint         - Lint code with Pylint"
	@echo "make test         - Run tests"
	@echo ""

setup:
	@echo "Setting up environment..."
	@mkdir -p data models metrics outputs logs
	@if [ ! -f .env ]; then cp .env.example .env; echo ".env created from .env.example"; fi
	@echo "Environment setup complete!"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed!"

install-dev:
	@echo "Installing dev dependencies..."
	pip install -r requirements.txt
	pip install black pylint pytest pytest-cov
	@echo "Dev dependencies installed!"

local:
	@echo "Running local pipeline..."
	python app.py

azure:
	@echo "Running pipeline on Azure ML..."
	python app.py --azure

demo:
	@echo "Running demo script..."
	python demo.py

clean:
	@echo "Cleaning output files..."
	rm -rf models/*.pkl models/*.joblib
	rm -rf metrics/*.json metrics/*.png
	rm -rf outputs/*
	rm -rf logs/*
	rm -rf __pycache__ .pytest_cache .coverage
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Removing all output and data..."
	rm -rf data/*
	rm -rf .env
	@echo "Full cleanup complete!"

format:
	@echo "Formatting code..."
	black . --line-length 100
	@echo "Code formatted!"

lint:
	@echo "Linting code..."
	pylint --disable=all --enable=E,F *.py
	@echo "Linting complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=. --cov-report=html
	@echo "Tests complete! See htmlcov/index.html for coverage report"

.DEFAULT_GOAL := help
