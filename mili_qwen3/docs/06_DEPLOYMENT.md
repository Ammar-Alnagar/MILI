# MILI Deployment Guide

## Overview

This guide covers deploying MILI inference system in production environments.

---

## Part 1: FastAPI Server Setup

### Server Architecture

**File: `python_layer/server/api.py`**

```python
"""FastAPI inference server for MILI."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json

from ..model.qwen3_model import Qwen3Model
from ..inference.scheduler import ContinuousBatchScheduler
from ..tokenizer.qwen_tokenizer import get_tokenizer


# Request/Response models
class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    stream: bool = False


class GenerationResponse(BaseModel):
    """Text generation response."""
    request_id: str
    generated_text: str
    tokens: List[int]
    total_tokens: int


# Initialize FastAPI
app = FastAPI(title="MILI Inference Server")

# Global state
model = None
scheduler = None
tokenizer = None


@app.on_event("startup")
async def startup():
    """Initialize model and scheduler on startup."""
    global model, scheduler, tokenizer
    
    from ..model.config import Qwen3Config
    from ..model.weight_loader import WeightLoader
    
    # Load configuration
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-7B")
    
    # Load weights
    weight_loader = WeightLoader(
        model_path="./models/Qwen3-7B",
        config=config,
        device="cuda"
    )
    weight_loader.load_from_huggingface("Qwen/Qwen3-7B")
    
    # Initialize model
    model = Qwen3Model(config, weight_loader)
    
    # Initialize scheduler
    scheduler = ContinuousBatchScheduler(
        max_batch_size=64,
        prefill_batch_size=32,
        decode_batch_size=256
    )
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    print("Model loaded and ready for inference")


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text from prompt.
    
    Args:
        request: GenerationRequest with prompt and parameters
        
    Returns:
        GenerationResponse with generated text
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize prompt
        tokens = tokenizer.encode(request.prompt)
        
        # Add to scheduler
        request_id = scheduler.add_request(
            prompt_tokens=tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # Generate (simplified - actual would run scheduler loop)
        generated = await generate_tokens(request_id, request.max_tokens)
        
        # Decode response
        generated_text = tokenizer.decode(generated)
        
        return GenerationResponse(
            request_id=request_id,
            generated_text=generated_text,
            tokens=generated,
            total_tokens=len(tokens) + len(generated)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get status of a generation request."""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = scheduler.get_request_status(request_id)
    return status


async def generate_tokens(request_id: str, max_tokens: int) -> List[int]:
    """Generate tokens for a request."""
    generated = []
    
    for _ in range(max_tokens):
        # Get next batch
        batch, batch_type = scheduler.get_next_batch()
        
        if not batch:
            await asyncio.sleep(0.01)
            continue
        
        # Run inference (simplified)
        # In real impl: call Mojo kernels
        
        await asyncio.sleep(0.001)
    
    return generated


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scheduler_ready": scheduler is not None
    }
```

---

## Part 2: Docker Deployment

### Dockerfile

**File: `deployment/docker/Dockerfile`**

```dockerfile
# Multi-stage build for MILI inference

# Stage 1: Build Mojo kernels
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

# Install Mojo
RUN curl https://docs.modular.com/mojo/install.sh | bash

# Copy kernel source
COPY mojo_kernels /app/mojo_kernels
WORKDIR /app/mojo_kernels

# Build kernels
RUN bash build.sh

# Stage 2: Runtime image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY python_layer /app/python_layer
COPY config /app/config
COPY models /app/models

# Copy compiled kernels from builder
COPY --from=builder /app/mojo_kernels/lib /app/mojo_kernels/lib

# Set environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "python_layer.server.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

**File: `deployment/docker/docker-compose.yml`**

```yaml
version: '3.8'

services:
  mili-server:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=Qwen/Qwen3-7B
      - MAX_BATCH_SIZE=64
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    gpus:
      - driver: nvidia
        count: 1
        capabilities: [compute, utility]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

---

## Part 3: Kubernetes Deployment

### Deployment YAML

**File: `deployment/kubernetes/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mili-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mili-inference
  template:
    metadata:
      labels:
        app: mili-inference
    spec:
      containers:
      - name: mili
        image: mili:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "32Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "48Gi"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_NAME
          value: "Qwen/Qwen3-7B"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      nodeSelector:
        gpu: "true"
```

### Service YAML

**File: `deployment/kubernetes/service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mili-inference
spec:
  selector:
    app: mili-inference
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deployment Commands

```bash
# Build image
docker build -t mili:latest -f deployment/docker/Dockerfile .

# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Check status
kubectl get pods
kubectl logs -f deployment/mili-inference

# Port forward for local testing
kubectl port-forward service/mili-inference 8000:80
```

---

## Part 4: Performance Optimization

### GPU Optimization

```python
"""GPU optimization utilities."""

import torch
import torch.cuda as cuda

def optimize_gpu():
    """Optimize GPU for inference."""
    # Use TF32 for better throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Disable grad computation
    torch.no_grad().__enter__()
    
    print("GPU optimizations applied")

def print_gpu_stats():
    """Print GPU memory statistics."""
    print(f"GPU memory allocated: {cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"GPU memory cached: {cuda.memory_cached() / 1e9:.2f} GB")
```

### Monitoring

```python
"""Monitoring and metrics collection."""

from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter('mili_requests_total', 'Total requests')
generation_time = Histogram('mili_generation_seconds', 'Generation time')
queue_size = Gauge('mili_queue_size', 'Current queue size')
gpu_memory = Gauge('mili_gpu_memory_bytes', 'GPU memory usage')

def record_generation(duration: float):
    """Record generation metric."""
    request_count.inc()
    generation_time.observe(duration)

def monitor_gpu():
    """Monitor GPU metrics."""
    import torch
    gpu_memory.set(torch.cuda.memory_allocated())
```

---

## Part 5: Load Testing

```python
"""Load testing utilities."""

import asyncio
import aiohttp
import time
from typing import List

async def load_test(
    url: str = "http://localhost:8000/generate",
    num_requests: int = 100,
    concurrent: int = 10
):
    """
    Run load test against inference server.
    
    Args:
        url: Server URL
        num_requests: Total number of requests
        concurrent: Concurrent requests
    """
    semaphore = asyncio.Semaphore(concurrent)
    
    async def make_request(session, prompt):
        async with semaphore:
            try:
                async with session.post(
                    url,
                    json={
                        "prompt": prompt,
                        "max_tokens": 100,
                        "temperature": 0.7
                    },
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    return await resp.json()
            except Exception as e:
                return {"error": str(e)}
    
    prompts = [f"Write a story about {i}" for i in range(num_requests)]
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        
        tasks = [make_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start
    
    # Analyze results
    successful = sum(1 for r in results if "error" not in r)
    failed = num_requests - successful
    
    print(f"Load Test Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Throughput: {num_requests/duration:.2f} req/s")
    print(f"  Average latency: {duration/num_requests*1000:.2f} ms")
```

---

## Part 6: Quick Start Commands

```bash
# Build locally
python -m pip install -r requirements.txt
mojo build -o lib/core/rope.so mojo_kernels/core/rope.🔥

# Run server locally
python -m uvicorn python_layer.server.api:app --reload --port 8000

# Test server
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Docker deployment
docker-compose -f deployment/docker/docker-compose.yml up

# Kubernetes deployment
kubectl apply -f deployment/kubernetes/

# Load testing
python -c "
import asyncio
from deployment.load_test import load_test
asyncio.run(load_test(num_requests=100, concurrent=10))
"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, enable quantization |
| Slow generation | Check GPU utilization, profile kernels |
| Server crashes | Check logs with `kubectl logs`, enable debug mode |
| High latency | Reduce max_tokens, use streaming response |

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/overview/working-with-objects/best-practices/)
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
