# Security Policy

## Supported Versions

The following versions of Omni-IPS are currently supported with security updates:

| Version | Supported          |
|---------|--------------------|
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in this repository, **please do not open a public GitHub Issue**.

### How to Report

Please use one of the following private channels:

1. **GitHub Private Vulnerability Reporting** (Preferred):
   - Go to the [Security tab](../../security/advisories/new) of this repository
   - Click **"Report a vulnerability"**
   - Fill in the vulnerability details

2. **Email** (Alternative):
   - Send details to the repository maintainer via GitHub profile contact
   - Subject line: `[SECURITY] Omni-IPS Vulnerability Report`

### What to Include

Please include the following information in your report:

- **Description**: A clear description of the vulnerability
- **Affected Component**: Which module/file/endpoint is affected (e.g., `api/main.py`, `/api/solve` endpoint)
- **Reproduction Steps**: Step-by-step instructions to reproduce the issue
- **Impact Assessment**: The potential impact (e.g., data exposure, RCE, DoS)
- **CVSS Score** (if applicable): Estimated severity using [CVSS v3.1](https://www.first.org/cvss/calculator/3.1)
- **Suggested Fix** (optional): If you have a proposed fix or mitigation
- **Environment**: Python version, OS, Docker version, relevant config

---

## Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial acknowledgement | Within **48 hours** |
| Vulnerability assessment | Within **5 business days** |
| Patch development | Within **14 business days** (critical: 7 days) |
| Public disclosure | After patch is released and users are notified |

---

## Vulnerability Severity Classification

We classify vulnerabilities using the CVSS v3.1 standard:

| Severity | CVSS Score | Response SLA |
|----------|------------|-------------|
| **Critical** | 9.0 - 10.0 | 7 days |
| **High** | 7.0 - 8.9 | 14 days |
| **Medium** | 4.0 - 6.9 | 30 days |
| **Low** | 0.1 - 3.9 | 60 days |
| **Informational** | 0.0 | Next release |

---

## Security Scope

### In Scope

The following components are within scope for security reports:

- **FastAPI REST endpoints** (`api/main.py`) — injection, auth bypass, DoS
- **Prolog inference engine** (`inference_engine/`) — malformed input, logic injection
- **Neo4j integration** (`knowledge_base/`) — Cypher injection, data leakage
- **Qdrant vector store** (`rag/`) — data poisoning, unauthorized access
- **Docker configuration** (`docker-compose.yml`, `Dockerfile`) — privilege escalation, secrets exposure
- **Environment variables / secrets** (`.env` handling) — credential leakage
- **GraphRAG pipeline** (`rag/graph_rag.py`) — prompt injection, LLM manipulation

### Out of Scope

- Vulnerabilities in third-party dependencies (report upstream to the respective project)
- Social engineering attacks
- Physical security issues
- Issues in forked or modified versions of this repository
- Denial of Service via resource exhaustion (unless a specific exploit is demonstrated)

---

## Security Best Practices for Deployment

When deploying Omni-IPS in production, follow these guidelines:

### Environment Variables
```bash
# Never commit .env files to version control
# Use secrets management (e.g., GitHub Secrets, HashiCorp Vault, AWS Secrets Manager)
NEO4J_PASSWORD=<use-strong-random-password>
OPENAI_API_KEY=<stored-in-secrets-manager>
QDRANT_API_KEY=<stored-in-secrets-manager>
```

### Network Security
- Run Neo4j and Qdrant on internal Docker network only (not exposed to public internet)
- Use a reverse proxy (Nginx/Traefik) with TLS termination in front of the FastAPI service
- Restrict Neo4j bolt port (`7687`) and Qdrant port (`6333`) to localhost or private network

### API Security
- Enable rate limiting on the FastAPI application
- Use API keys or JWT authentication for production deployments
- Validate and sanitize all inputs to `/solve`, `/api/solve`, and `/api/explain` endpoints

### Docker Security
- Run containers as non-root users
- Use read-only filesystem mounts where possible
- Regularly update base images

---

## Disclosure Policy

This project follows **Coordinated Vulnerability Disclosure (CVD)**:

1. Reporter submits vulnerability privately
2. Maintainer acknowledges and investigates
3. Fix is developed and tested
4. Security advisory is published on GitHub with CVE (if applicable)
5. Reporter is credited (unless they prefer anonymity)

---

## Security Contact

Maintainer: **[@khang3004](https://github.com/khang3004)**

For sensitive security communications, please use the [GitHub Private Vulnerability Reporting](../../security/advisories/new) feature.

---

## Acknowledgements

We sincerely thank all security researchers who responsibly disclose vulnerabilities and help improve the security of Omni-IPS.
