# Security Policy

## Supported Versions

The following versions of **AlphaGeometry-IMO** are currently supported with security updates:

| Version | Supported          |
|---------|--------------------|
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of AlphaGeometry-IMO seriously. If you discover a security vulnerability in this repository, **please do not open a public GitHub Issue**.

### How to Report

Please use one of the following private reporting channels:

1. **GitHub Private Vulnerability Reporting** (Preferred):
   - Go to the [Security tab](../../security/advisories/new) of this repository.
   - Click **"Report a vulnerability"**.
   - Fill in the details.

2. **Email Contact** (Alternative):
   - Contact the maintainer via GitHub profile details.
   - Subject line: `[SECURITY] AlphaGeometry-IMO Vulnerability Report`

### Information to Include

Please include:
- **Description**: Clear summary of the vulnerability.
- **Affected Component**: Which module/file/endpoint is affected (e.g., `api/main.py`, `/geo/solve` endpoint, `rag_agent/router.py`).
- **Reproduction Steps**: Step-by-step instructions to reproduce the issue.
- **Impact Assessment**: Potential severity (e.g., Prompt Injection, Denial of Service, Credential Leakage).
- **Environment**: Python version, OS, Docker version, relevant `.env` configuration.

---

## Response SLA Timeline

| Stage | SLA Timeline |
|-------|--------------|
| Initial Acknowledgement | Within **48 hours** |
| Vulnerability Assessment | Within **5 business days** |
| Patch Development | Within **14 business days** (Critical: 7 days) |
| Public Disclosure | After patch release and user notification |

---

## Security Scope

### In Scope

The following components are within scope for security reports:

- **FastAPI REST Endpoints** (`api/main.py`) — Input validation, endpoint security, rate limiting, DoS protection.
- **Symbolic Core & Unifier** (`core_engine/`) — Malformed predicate input handling, recursion limits.
- **Auxiliary Agent & LLM Router** (`geo_engine/auxiliary_agent.py`, `rag_agent/router.py`) — Prompt injection, unverified auxiliary construction validation.
- **Neo4j Integration** (`graph_db/connection.py`) — Cypher query sanitization, parameter binding.
- **Qdrant Vector DB** (`graph_db/qdrant_factory.py`) — Vector payload filtering, authorization.
- **Docker Infrastructure** (`docker-compose.yml`, `Dockerfile`) — Container privilege isolation, environment secret handling.
- **Environment Variables** (`.env` handling) — Sensitive API key exposure (`GEMINI_API_KEY`, `NEO4J_PASSWORD`).

### Out of Scope

- Vulnerabilities in third-party dependencies (report directly to upstream maintainers).
- Social engineering or physical security attacks.
- Issues in modified or uncommitted local forks.

---

## Security Best Practices for Production Deployment

### 1. Secret & Key Management
```bash
# Never commit .env files to version control
# Use dedicated secrets management (GitHub Secrets, Vault, Cloud KMS)
GEMINI_API_KEY=<stored-in-secrets-manager>
NEO4J_PASSWORD=<use-strong-random-password>
```

### 2. Network Isolation
- Restrict Neo4j Bolt (`7687`) and Qdrant (`6333`) ports to local container networks or private VPCs.
- Deploy a reverse proxy (e.g. Traefik / Nginx) with TLS termination in front of `api/main.py`.

### 3. API Input Sanitization
- Validate all incoming natural language and predicate inputs at the FastAPI model layer.
- Enforce strict timeouts on LLM API calls and symbolic solver iterations.

---

## Security Contact

Maintainer: **[@khang3004](https://github.com/khang3004)**
