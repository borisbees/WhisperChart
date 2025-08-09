# WhisperChart Makefile
# Usage: run any of these in your repo root, e.g.:
#   make venv     # set up the Python environment
#   make install  # install requirements
#   make run      # run Streamlit app
#   make jupyter  # open Jupyter notebook (if installed)
#   make clean    # remove the venv and __pycache__

.PHONY: venv install run jupyter clean lint format dev-install precommit

venv:
	@echo "🔧 Creating Python virtual environment..."
	python3 -m venv venv

install:
	@echo "📦 Activating venv and installing dependencies..."
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	@echo "🚀 Running Streamlit app..."
	. venv/bin/activate && streamlit run app/app.py

jupyter:
	@echo "🧪 Launching Jupyter Notebook..."
	. venv/bin/activate && jupyter notebook notebooks/whisperchart_dev.ipynb

clean:
	@echo "🧹 Removing virtual environment and Python cache files..."
	rm -rf venv __pycache__ */__pycache__

help:
	@echo "Available targets:"
	@echo "  venv     - Set up Python virtual environment"
	@echo "  install  - Install all requirements"
	@echo "  run      - Run Streamlit app"
	@echo "  jupyter  - Open Jupyter notebook"
	@echo "  clean    - Remove venv and cache"
	@echo "  lint     - Run Ruff and Black checks"
	@echo "  format   - Autoformat with Black and Ruff"
	@echo "  dev-install - Install dev tools (ruff/black/pre-commit)"
	@echo "  precommit  - Install pre-commit git hooks"

lint:
	@echo "🔎 Linting with Ruff and Black..."
	. venv/bin/activate && ruff check $(shell git ls-files "*.py") && black --check $(shell git ls-files "*.py")

format:
	@echo "🧼 Formatting with Black and Ruff..."
	. venv/bin/activate && black $(shell git ls-files "*.py") && ruff check --fix $(shell git ls-files "*.py")

dev-install:
	@echo "🛠 Installing dev tools (ruff, black, pre-commit)..."
	. venv/bin/activate && pip install -r requirements-dev.txt || pip install ruff black pre-commit

precommit:
	@echo "🔗 Installing pre-commit hooks..."
	. venv/bin/activate && pre-commit install
