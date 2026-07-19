# Operation & Reproduction Guide

This document provides step-by-step instructions to set up the environment, run core engine verifications, execute data ingestion pipelines, and reproduce benchmark results for **AlphaGeometry-IMO**.

---

## 1. System Requirements

- **Python**: Python `3.10` or higher (tested on Python 3.12).
- **Package Manager**: Astral `uv` (Rust-based ultra-fast package manager).
- **Orchestration Tool**: `make` (Native on macOS/Linux; available via Git Bash/WSL on Windows).
- **Database Services** (Optional for local dry-runs; required for GraphRAG):
  - **Neo4j**: `5.x` Community/Enterprise (Bolt port `7687`, HTTP port `7474`).
  - **Qdrant**: `1.x` Vector Database (HTTP port `6333`).

---

## 2. Quick Setup

### Step 1: Environment Initialization
Clone the repository and run the setup script:
```bash
make setup
```
This command automatically installs `uv` (if not present), creates a virtual environment in `.venv/`, and syncs all dependencies.

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys (optional if using offline fallback mode):
```bash
cp .env.example .env
```
Key configuration parameters:
- `GEMINI_API_KEY`: API key for LLM-based auxiliary construction agent and natural language parser.
- `NEO4J_URI`: Neo4j database endpoint (default: `bolt://localhost:7687`).
- `NEO4J_PASSWORD`: Neo4j password (default: `geo_ips_password`).
- `QDRANT_HOST`: Qdrant endpoint (default: `localhost`).

---

## 3. Running Test Suites & Benchmark Reproduction

### 3.1. Verification 1: Core Engine & Unifier Unit Tests
Verify that symbolic forward-chaining, backward-chaining, unification, and commutative canonicalization operate correctly:
```bash
python3 tests/verify_scaffold.py
```
**Expected Output:** `🎉 ALL ALPHAGEOMETRY CORE SOLVER VERIFICATIONS PASSED SUCCESSFULLY!`

### 3.2. Verification 2: Official IMO & Curriculum Benchmark Suite (`geometry_test_suite.md`)
Execute all 15 official benchmark problems defined in `geometry_test_suite.md`:
```bash
python3 tests/run_geometry_suite.py
```
This runner evaluates problem statements across 4 difficulty tiers:
1. **Basic Tier**: Triangle Angle Sum, SAS Congruence, Parallel Line Corresponding Angles.
2. **Intermediate Tier**: Cyclic Quadrilateral Opposite Angles, Midpoint Proportion, Intersecting Chords Theorem.
3. **Advanced Tier**: Right Triangle Metric Height Relation, Ptolemy's Theorem, Tangent-Secant Theorem, Rhombus Diagonals.
4. **Olympiad Tier**: Ceva's Theorem, Menelaus's Theorem, Simson Line Theorem, Varignon's Midpoint Theorem, Nagel Point Precursor.

---

## 4. Knowledge Base Ingestion Pipelines

To populate Neo4j and Qdrant with the full library of Euclidean geometry axioms, FormalGeo rules, and OWL ontology:

1. **Initialize Database Schemas & Constraints:**
   ```bash
   make setup-schema
   ```

2. **Ingest Fundamental Euclidean Axioms:**
   ```bash
   make ingest-geometry
   ```

3. **Ingest Expert AlphaGeometry & FormalGeo Theorems:**
   ```bash
   make ingest-expert
   make ingest-formalgeo
   ```

4. **Ingest OWL Geometry Class Hierarchy:**
   ```bash
   make ingest-ontology
   ```

5. **Single-Command Full Ingestion:**
   ```bash
   make ingest-all
   ```

---

## 5. Running Interactive Services

### 5.1. FastAPI REST Server
Start the local REST API server:
```bash
make run-server
```
Interactive OpenAPI documentation will be available at:
- **Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

### 5.2. Streamlit Web UI
In a separate terminal, start the Streamlit web application:
```bash
make run-ui
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 6. Docker Container Deployment

To launch the full containerized stack (Neo4j, Qdrant, FastAPI backend, and Streamlit UI):

```bash
make docker-up
```

To monitor status:
```bash
make docker-status
make docker-logs
```

To stop all services:
```bash
make docker-down
```

---

## 7. Workspace Cleanup

To purge temporary bytecode, caches, and virtual environments:
```bash
make clean
```
