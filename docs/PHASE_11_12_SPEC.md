# Phase 11 & 12 Spec — AWS Cloud Architecture & Continuous Deployment

## Overview
Phases 11 & 12 transition ApexGuard from local docker-compose execution to automated AWS cloud hosting using EC2, S3, ECR, CloudWatch, and GitHub Actions CD automation.

## Cloud Architecture

```text
                                     AWS Cloud
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                 │
    AWS S3 (Document Store)                                       AWS ECR (Docker Registry)
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
                                   AWS EC2 Instance
                             (Docker Compose Engine)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
          FastAPI Gateway                                Qdrant / Redis
```

## Setup Specification

1. **AWS ECR:** Container registry repository `apexguard/api`.
2. **AWS S3:** Bucket `apexguard-knowledge-docs` for raw document storage.
3. **AWS EC2:** Ubuntu t3.medium / g4dn instance configured with Docker engine.
4. **AWS IAM:** Least-privilege role granting EC2 read/write access to S3 and pull access to ECR.
5. **Continuous Deployment Workflow:** On git push to `main`, GitHub Actions builds the image, pushes to ECR, and executes an SSH remote script on EC2 to perform zero-downtime rolling container updates.
