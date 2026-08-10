# Phase 10 Spec — Multi-Container Docker & GitHub Actions CI/CD

## Overview
Phase 10 standardizes the containerization of all ApexGuard services via Docker Compose and configures a GitHub Actions pipeline for automated linting, test execution, and image building.

## Docker Services Stack (`docker-compose.yml`)

- `apexguard-api`: FastAPI Gateway container.
- `qdrant`: Vector database service.
- `redis`: Redis cache & session store.
- `ollama`: LLM serving container.

## GitHub Actions CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: ApexGuard CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install dependencies
        run: pip install -r requirements.txt pytest
      - name: Run Test Suite
        run: pytest tests/
      - name: Build Docker Image
        run: docker build -t apexguard-api:latest .
```
