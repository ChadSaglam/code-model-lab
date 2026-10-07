PYTHON := .venv/bin/python

.PHONY: help setup check dev

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[1m%-8s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install the pinned requirements
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

check: ## Run every check; stop on the first failure
	$(PYTHON) -m pytest -q

dev: ## Start the lab at http://127.0.0.1:8765
	$(PYTHON) lab/serve_lab.py
