# Phase 6 Spec — Redis State, Memory & Response Caching

## Overview
Phase 6 adds high-speed in-memory state management using Redis for conversation history sliding windows, session isolation, distributed rate-limiting buckets, and semantic response caching.

## Redis Data Layout

```text
Keyspace Schema:
  apexguard:session:{session_id}:history    -> LIST (JSON serialized message turns)
  apexguard:cache:response:{query_hash}     -> STRING (Cached response payload with TTL)
  apexguard:ratelimit:{api_key}             -> HASH (Token bucket counters)
```

## Features

1. **Exact & Semantic Caching:**
   - Hash-based exact string matching for instant (< 5ms) cached query resolution.
   - Optional vector-similarity cache lookup over recent historical queries.

2. **Session Memory Windowing:**
   - Maintains last $N$ turns per conversation session ID with automatic TTL expiration (e.g., 24h).
