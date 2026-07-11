# ==============================================================================
#                GeoIPS — Plane Geometry Intelligent Problem Solver
#                         Unified Operational Makefile
# ==============================================================================
# Powered by Astral 'uv' Package Manager & Docker Containerization
# ==============================================================================

.PHONY: help setup setup-schema test test-unifier test-rag ingest-geometry ingest-expert ingest-formalgeo ingest-ontology ingest-all embed-knowledge docker-up docker-down docker-logs docker-status run-server run-ui clean

SHELL := /bin/bash
UV    := $(shell which uv 2>/dev/null)

COLOR_RESET  = \033[0m
COLOR_BOLD   = \033[1m
COLOR_GREEN  = \033[32m
COLOR_BLUE   = \033[34m
COLOR_CYAN   = \033[36m
COLOR_YELLOW = \033[33m
COLOR_RED    = \033[31m

PORT ?= 8080
HOST ?= 0.0.0.0

help:
	@echo -e "$(COLOR_BOLD)$(COLOR_CYAN)========================================================================$(COLOR_RESET)"
	@echo -e "$(COLOR_BOLD)$(COLOR_GREEN)          GeoIPS — Plane Geometry IPS Command Panel (uv-Powered)      $(COLOR_RESET)"
	@echo -e "$(COLOR_BOLD)$(COLOR_CYAN)========================================================================$(COLOR_RESET)"
	@echo -e "$(COLOR_BOLD)Environment Status:$(COLOR_RESET)"
	@if [ -z "$(UV)" ]; then \
		echo -e "  uv Package Manager: $(COLOR_RED)NOT FOUND$(COLOR_RESET)"; \
	else \
		echo -e "  uv Package Manager: $(COLOR_GREEN)FOUND$(COLOR_RESET) ($(shell uv --version))"; \
	fi
	@echo -e ""
	@echo -e "$(COLOR_BOLD)1. Setup:$(COLOR_RESET)"
	@echo -e "  $(COLOR_CYAN)make setup$(COLOR_RESET)              - Install dependencies via uv"
	@echo -e "  $(COLOR_CYAN)make setup-schema$(COLOR_RESET)       - Initialize Neo4j constraints"
	@echo -e "  $(COLOR_CYAN)make clean$(COLOR_RESET)              - Remove caches and .venv"
	@echo -e ""
	@echo -e "$(COLOR_BOLD)2. Testing:$(COLOR_RESET)"
	@echo -e "  $(COLOR_CYAN)make test-unifier$(COLOR_RESET)       - Run unifier & solver unit tests (Phase 3)"
	@echo -e "  $(COLOR_CYAN)make test$(COLOR_RESET)               - Run all pytest tests"
	@echo -e "  $(COLOR_CYAN)make test-rag$(COLOR_RESET)           - Run GraphRAG pipeline integration tests"
	@echo -e ""
	@echo -e "$(COLOR_BOLD)3. Knowledge Base Ingestion:$(COLOR_RESET)"
	@echo -e "  $(COLOR_CYAN)make ingest-geometry$(COLOR_RESET)    - Ingest Euclidean geometry theorems + variable rules"
	@echo -e "  $(COLOR_CYAN)make ingest-expert$(COLOR_RESET)      - Ingest AlphaGeometry/FormalGeo expert theorems"
	@echo -e "  $(COLOR_CYAN)make ingest-formalgeo$(COLOR_RESET)   - Ingest 196+ FormalGeo logic rules"
	@echo -e "  $(COLOR_CYAN)make ingest-ontology$(COLOR_RESET)    - Load OWL class hierarchy into Neo4j"
	@echo -e "  $(COLOR_CYAN)make ingest-all$(COLOR_RESET)         - Run all ingestion pipelines (geometry + expert + formalgeo + ontology)"
	@echo -e "  $(COLOR_CYAN)make embed-knowledge$(COLOR_RESET)    - Populate Qdrant from Neo4j"
	@echo -e ""


	@echo -e "$(COLOR_BOLD)4. Run:$(COLOR_RESET)"
	@echo -e "  $(COLOR_CYAN)make run-server$(COLOR_RESET)         - Launch FastAPI server on http://$(HOST):$(PORT)"
	@echo -e "  $(COLOR_CYAN)make run-ui$(COLOR_RESET)             - Launch Streamlit UI on http://localhost:8501"
	@echo -e ""
	@echo -e "$(COLOR_BOLD)5. Docker:$(COLOR_RESET)"
	@echo -e "  $(COLOR_CYAN)make docker-up$(COLOR_RESET)          - Build & start Neo4j + Backend + Frontend"
	@echo -e "  $(COLOR_CYAN)make docker-down$(COLOR_RESET)        - Shut down all containers"
	@echo -e "  $(COLOR_CYAN)make docker-status$(COLOR_RESET)      - View container status"
	@echo -e "  $(COLOR_CYAN)make docker-logs$(COLOR_RESET)        - Tail container logs"
	@echo -e "$(COLOR_BOLD)$(COLOR_CYAN)========================================================================$(COLOR_RESET)"

setup:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Setting up Python environment via uv...$(COLOR_RESET)"
	@if [ -z "$(UV)" ]; then \
		echo -e "$(COLOR_YELLOW)[WARNING] 'uv' not found. Installing...$(COLOR_RESET)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
	fi
	@uv sync
	@echo -e "$(COLOR_GREEN)[GeoIPS] Setup complete. Virtual environment ready in '.venv/'.$(COLOR_RESET)"

setup-schema:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Initializing Neo4j constraints...$(COLOR_RESET)"
	@uv run python data_pipelines/setup_schema.py

test:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Running all pytest tests...$(COLOR_RESET)"
	@uv run pytest tests/ -v

test-unifier:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Running Unifier & Variable Binding tests (Phase 3)...$(COLOR_RESET)"
	@uv run pytest tests/test_unifier.py -v

test-rag:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Running GraphRAG pipeline integration tests...$(COLOR_RESET)"
	@uv run python tests/verify_rag.py

ingest-geometry:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Ingesting Euclidean geometry theorems + variable rules...$(COLOR_RESET)"
	@uv run python data_pipelines/ingest_geometry.py
	@echo -e "$(COLOR_GREEN)[GeoIPS] Geometry ingestion complete.$(COLOR_RESET)"

ingest-expert:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Ingesting AlphaGeometry/FormalGeo expert theorems...$(COLOR_RESET)"
	@uv run python data_pipelines/ingest_expert_rules.py
	@echo -e "$(COLOR_GREEN)[GeoIPS] Expert rules ingestion complete.$(COLOR_RESET)"

ingest-formalgeo:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Ingesting FormalGeo GDL theorems into Neo4j + Qdrant Cloud...$(COLOR_RESET)"
	@uv run python data_pipelines/ingest_formalgeo_rules.py
	@echo -e "$(COLOR_GREEN)[GeoIPS] FormalGeo theorems ingestion complete.$(COLOR_RESET)"

ingest-ontology:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Loading OWL geometry ontology into Neo4j...$(COLOR_RESET)"
	@uv run python data_pipelines/ingest_ontology.py
	@echo -e "$(COLOR_GREEN)[GeoIPS] Ontology ingestion complete.$(COLOR_RESET)"

ingest-all:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Running all ingestion pipelines...$(COLOR_RESET)"
	@uv run python data_pipelines/setup_schema.py
	@uv run python data_pipelines/ingest_geometry.py
	@uv run python data_pipelines/ingest_expert_rules.py
	@uv run python data_pipelines/ingest_formalgeo_rules.py
	@uv run python data_pipelines/ingest_ontology.py
	@echo -e "$(COLOR_GREEN)[GeoIPS] All ingestion pipelines complete.$(COLOR_RESET)"



embed-knowledge:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Populating Qdrant Vector DB from Neo4j...$(COLOR_RESET)"
	@uv run python rag_agent/embed_knowledge.py

run-server:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Launching FastAPI server at http://$(HOST):$(PORT)...$(COLOR_RESET)"
	@uv run uvicorn api.main:app --reload --host $(HOST) --port $(PORT)

run-ui:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Launching Streamlit UI at http://localhost:8501...$(COLOR_RESET)"
	@uv run streamlit run ui/app.py --server.port 8501

docker-up:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Starting containerized infrastructure (Neo4j + Backend + Frontend)...$(COLOR_RESET)"
	docker compose up --build -d
	@echo -e "$(COLOR_GREEN)[GeoIPS] Infrastructure launched.$(COLOR_RESET)"
	@echo -e "  - Neo4j Browser: http://localhost:7474  (neo4j / geo_ips_password)"
	@echo -e "  - FastAPI docs:  http://localhost:8080/docs"
	@echo -e "  - Streamlit UI:  http://localhost:8501"

docker-down:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Shutting down containers...$(COLOR_RESET)"
	docker compose down
	@echo -e "$(COLOR_GREEN)[GeoIPS] All containers stopped.$(COLOR_RESET)"

docker-status:
	docker compose ps

docker-logs:
	docker compose logs -f

clean:
	@echo -e "$(COLOR_BLUE)[GeoIPS] Cleaning caches and build artifacts...$(COLOR_RESET)"
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .venv .uv .pytest_cache
	@echo -e "$(COLOR_GREEN)[GeoIPS] Cleanup complete.$(COLOR_RESET)"
