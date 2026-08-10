# Phase 9 Spec — AI Guardrails & Security Gateway

## Overview
Phase 9 constructs security guardrails across inbound prompts and outbound responses to prevent PII leaks, block prompt injection attacks, and enforce request boundaries.

## Architecture

```text
User Request -> PII Sanitizer Middleware -> Prompt Injection Classifier -> RAG / LLM -> Output Guardrails -> Client Response
```

## Security Controls

1. **PII Detection & Redaction:**
   - Detects and masks Email addresses, Phone numbers, Social Security Numbers, Credit Card numbers, and API Keys using regex patterns.
   - Example: `john.doe@email.com` → `[REDACTED_EMAIL]`.

2. **Prompt Injection Defenses:**
   - Filters common jailbreak heuristics ("Ignore previous instructions", "System override").

3. **Request Boundaries:**
   - Enforces max body payload limits (e.g., 2MB) and strict tool execution whitelists.
