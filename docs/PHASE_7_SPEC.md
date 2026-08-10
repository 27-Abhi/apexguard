# Phase 7 Spec — Agentic Tool Calling & Execution Engine

## Overview
Phase 7 equips the LLM Gateway with deterministic tool calling capabilities, strict Pydantic argument parsing, tool execution sandboxing, and multi-step agent reasoning loops.

## Registered Tools Schema

1. `search_knowledge(query: str, top_k: int = 3)` — Invokes the Phase 2 Hybrid RAG engine.
2. `get_document(filename: str)` — Retrieves complete text of a specific uploaded file.
3. `calculator(expression: str)` — Safe Python mathematical expression evaluator.
4. `external_api(endpoint: str, params: dict)` — Fetches live external HTTP API data.

## Execution Loop

```text
User Input -> Agent Router -> LLM Tool Call Request -> Schema Validation -> Tool Execution -> Result Feeding -> Final Answer Generation
```
