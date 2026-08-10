# Phase 15 & 16 Spec — Multimodal Voice RAG Integration & GCP Infrastructure Mapping

## Phase 15 Overview: Real-Time Multimodal Voice Pipeline
Phase 15 connects real-time audio streams over WebSockets to turn ApexGuard into an interactive Voice AI Platform.

```text
Audio Input Stream (Mic) ──> WebSocket ──> Whisper ASR ──> ApexGuard Gateway (RAG/Tools) ──> TTS Engine ──> Audio Output Stream
```

### Key Technical Handles:
- **WebSocket Gateway Endpoint:** `/ws/v1/voice`.
- **Low-Latency Streaming:** Audio chunk buffering, VAD (Voice Activity Detection), and concurrent TTS audio chunk synthesis.

---

## Phase 16 Overview: GCP Cloud Infrastructure Mapping
Phase 16 provides an architectural translation mapping for multi-cloud parity between AWS and Google Cloud Platform (GCP).

### Multi-Cloud Service Translation Matrix

| AWS Service | GCP Equivalent | ApexGuard Responsibility |
| :--- | :--- | :--- |
| **AWS EC2** | Compute Engine | Container Host Compute |
| **AWS S3** | Cloud Storage | Document & Vector Snapshot Store |
| **AWS ECR** | Artifact Registry | Container Image Repository |
| **AWS EKS** | GKE (Google Kubernetes Engine) | Kubernetes Cluster |
| **AWS CloudWatch** | Cloud Logging & Monitoring | Centralized Telemetry & Log Aggregation |
