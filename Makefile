.PHONY: install test lint bench run clean docker-build

PYTHON ?= python3

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .
	mypy src/

format:
	ruff format .
	ruff check --fix .

bench:
	$(PYTHON) scripts/benchmark.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache build dist *.egg-info data/*.duckdb
