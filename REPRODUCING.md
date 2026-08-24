# 🧪 AlphaGeometry-IMO: Reproduction & Verification Manual

This document provides definitive, end-to-end instructions to set up the environment, run core logic unit tests, execute the knowledge ingestion pipelines, reproduce the **15/15 IMO Benchmark Suite**, and compile the academic LaTeX report.

---

## 📋 1. System Requirements

| Component | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| **Operating System** | macOS (Apple Silicon/Intel), Ubuntu 22.04+, Windows WSL2 | macOS / Linux Ubuntu 24.04 |
| **Python** | Python `3.10` | Python `3.12` |
| **Package Manager** | Astral `uv` (>= 0.4.0) | Latest `uv` via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Containers** | Docker Engine `24.0+` & Docker Compose `v2+` | Docker Desktop / OrbStack |
| **Memory (RAM)** | 4 GB | 8 GB+ (for running Neo4j & Qdrant locally) |

---

## ⚡ 2. Quick Reproduction (Docker Method - Recommended)

The fastest and most reliable way to reproduce the entire environment with all databases, backend engines, and web UI pre-configured:

```bash
# 1. Clone repository
git clone https://github.com/khang3004/Knowledge_Representation_Playground_code.git
cd Knowledge_Representation_Playground_code

# 2. Launch containerized stack
make docker-up
```

### Verified Service Endpoints:
- 🖥️ **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
- ⚡ **FastAPI Swagger API**: [http://localhost:8080/docs](http://localhost:8080/docs)
- 🔷 **Neo4j Graph Database**: [http://localhost:7474](http://localhost:7474) *(User: `neo4j`, Password: `geo_ips_password`)*
- 🔴 **Qdrant Vector Engine**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

To monitor health and inspect logs:
```bash
make docker-status
make docker-logs
```

To gracefully shut down containers:
```bash
make docker-down
```

---

## 🛠️ 3. Local Python Environment Setup (`uv`)

If you prefer running directly in a local Python environment without Docker containers:

### Step 1: Initialize Virtual Environment
```bash
make setup
```
*This command checks for `uv`, automatically creates an isolated virtual environment in `.venv/`, and syncs all dependencies defined in `pyproject.toml` and `uv.lock`.*

### Step 2: Configure Environment Keys (Optional)
```bash
cp .env.example .env
```
Configure your preferred LLM provider for auxiliary construction and GraphRAG:
```dotenv
# Provider options: groq (fastest), gemini, openai
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Neo4j & Qdrant settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=geo_ips_password
QDRANT_HOST=localhost
QDRANT_PORT=6333
```
*(Note: The system contains built-in offline heuristic fallbacks for rule matching and theorem proving even without an active internet connection).*

---

## 🧪 4. Step-by-Step Test Verification

### Phase 1: Core Symbolic Deduction & Unification Engine
Verify that structural pattern matching, unification with variable bindings, canonical normalization ($AB \equiv BA$, $\angle ABC \equiv \angle CBA$), and forward chaining work with 100% precision:

```bash
python3 tests/verify_scaffold.py
```

**Expected Console Output:**
```text
[INFO] Testing Unification with variable bindings...
[INFO] Match found: {?x: 'A', ?y: 'B', ?z: 'C'}
[INFO] Testing Canonical Equivalence...
[INFO] Segment(B, A) == Segment(A, B) -> PASS
[INFO] Testing Forward Chaining Engine...
[INFO] Goal Equal(Angle(C), 50) reached in 3 steps!
======================================================================
🎉 ALL ALPHAGEOMETRY CORE SOLVER VERIFICATIONS PASSED SUCCESSFULLY!
======================================================================
```

---

### Phase 2: Official 15/15 IMO Benchmark Suite Evaluation
Execute the full benchmark suite defined in `geometry_test_suite.md`:

```bash
python3 tests/run_geometry_suite.py
```

**Benchmark Results:**
- **Tier 1 (Basic)**: 3/3 PASS (Triangle Angle Sum, SAS Congruence, Corresponding Angles).
- **Tier 2 (Intermediate)**: 3/3 PASS (Cyclic Quad Opposite Angles, Midpoint Segment, Intersecting Chords).
- **Tier 3 (Advanced)**: 4/4 PASS (Right Triangle Height Metric, Ptolemy's Theorem, Tangent-Secant Theorem, Rhombus Diagonals).
- **Tier 4 (Olympiad)**: 5/5 PASS (Ceva's Theorem, Menelaus's Theorem, Simson Line, Varignon's Theorem, Nagel Point).
- **Overall Success Rate**: **15/15 (100.0%)** with average latency ~8.7s.

---

### Phase 3: Knowledge Base & Vector Ingestion Pipelines
To populate the knowledge graph on Neo4j and vector index on Qdrant:

```bash
# 1. Initialize constraints and collections
make setup-schema

# 2. Ingest Euclidean axioms, FormalGeo theorems, and OWL ontology
make ingest-all
```

---

## 🖥️ 5. Running the Interactive UI & Services

### Launch FastAPI Solver Gateway:
```bash
make run-server
```
Test with curl:
```bash
curl -X POST "http://localhost:8080/geo/solve" \
  -H "Content-Type: application/json" \
  -d '{"query": "Cho tam giác ABC có góc A = 60, góc B = 70. Tính góc C."}'
```

### Launch Streamlit Chat Interface:
```bash
make run-ui
```
Open [http://localhost:8501](http://localhost:8501) to interact with the pedagogical proof generator and visual theorem graph explorer.

---

## 📄 6. Compiling the Academic LaTeX Report

The repository includes a 43-page comprehensive Vietnamese academic essay report conforming to HCMUS graduate format:

```bash
cd latex_report
latexmk -pdf main.tex
```

- **Output File**: [`latex_report/main.pdf`](./latex_report/main.pdf)
- **Features**: Vector TikZ Sacred Geometry cover emblem, clear architecture diagrams, step-by-step proofs, and BibTeX citations.

---

## 🧹 7. Workspace Cleanup

To purge cached files, Python bytecode, and temporary logs:
```bash
make clean
```
