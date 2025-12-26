# Python Integration Guide for MILI

## Python Integration

This guide covers the Python components of MILI, including the FastAPI server, HuggingFace transformers integration, and API design.

### Architecture

```
┌─────────────────────────────────┐
│      FastAPI Server             │  <- REST API endpoints
│      (server.py)                │
└────────────────────┬────────────┘
                     │
┌────────────────────▼────────────┐
│  HuggingFace Transformers       │  <- Model & tokenizer
│  AutoTokenizer + AutoModel      │
└────────────────────┬────────────┘
                     │
┌────────────────────▼────────────┐
│   Request Processing            │  <- Input validation
│   Pydantic Models               │
└────────────────────┬────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────▼──┐            ┌─────┬─┘
    │ GPU Inference     │ CPU Fallback
    │ (CUDA)            │ (PyTorch CPU)
    └────────┘           └──────────
```

---

## Server Implementation

### Main Server File

**File: `server.py`**

```python
#!/usr/bin/env python3
"""
MILI Qwen3 Inference Server
Standalone server script to serve the Qwen3 model.
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Request/Response models
class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50


class GenerationResponse(BaseModel):
    """Text generation response."""
    generated_text: str
    tokens: List[int]
    total_tokens: int
    prompt_tokens: int


# Initialize FastAPI
app = FastAPI(title="MILI Qwen3 Inference Server", version="0.1.0")

# Global state
model = None
tokenizer = None


@app.on_event("startup")
async def startup():
    """Initialize model and tokenizer on startup."""
    global model, tokenizer

    try:
        print(" Initializing Qwen3 Inference Server...")

        # Load model and tokenizer from HuggingFace
        model_path = "Qwen/Qwen3-0.6B"
        print(f"Loading model from: {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(" Tokenizer loaded")

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        print(" Model loaded")

        print(f" Model: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden, {model.config.vocab_size} vocab")

    except Exception as e:
        print(f" Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text from prompt.

    Args:
        request: GenerationRequest with prompt and parameters

    Returns:
        GenerationResponse with generated text
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize prompt
        inputs = tokenizer(request.prompt, return_tensors="pt")
        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"][0].tolist()
        print(f" Prompt: '{request.prompt}' ({len(prompt_tokens)} tokens)")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_tokens = outputs[0].tolist()
        generated_text = tokenizer.decode(full_tokens[len(prompt_tokens):], skip_special_tokens=True)

        print(
            f"🤖 Generated: '{generated_text}' ({len(full_tokens) - len(prompt_tokens)} tokens)"
        )

        return GenerationResponse(
            generated_text=generated_text,
            tokens=full_tokens,
            total_tokens=len(full_tokens),
            prompt_tokens=len(prompt_tokens),
        )

    except Exception as e:
        print(f" Generation error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_ready": tokenizer is not None,
        "model_type": "Qwen3-0.6B",
        "vocab_size": model.config.vocab_size if model else None,
        "max_seq_length": model.config.max_position_embeddings if model else None,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MILI Qwen3 Inference Server",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate (POST)",
    }


if __name__ == "__main__":
    import uvicorn

    print(" Starting MILI Qwen3 Inference Server...")
    uvicorn.run(app, host="0.0.0.0", port=9999)
```

## API Usage Examples

### Basic Text Generation

```python
import requests

# Generate a response
response = requests.post("http://localhost:9999/generate", json={
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
})

result = response.json()
print("Generated:", result["generated_text"])
print("Tokens used:", result["total_tokens"])
```

### Advanced Parameters

```python
# Creative writing
response = requests.post("http://localhost:9999/generate", json={
    "prompt": "Write a haiku about coding:",
    "max_tokens": 50,
    "temperature": 1.2,  # More creative
    "top_p": 0.9         # Nucleus sampling
})

# Technical explanation
response = requests.post("http://localhost:9999/generate", json={
    "prompt": "Explain neural networks simply:",
    "max_tokens": 150,
    "temperature": 0.3,  # More focused
    "top_k": 40          # Limit token selection
})
```

### Health Monitoring

```python
# Check server status
health = requests.get("http://localhost:9999/health").json()
print(f"Status: {health['status']}")
print(f"Model: {health['model_type']}")
print(f"Vocab size: {health['vocab_size']}")
```

## Direct Python Usage

For programmatic use without the server:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (same as server)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Configuration

### Generation Parameters

- **max_tokens**: Maximum tokens to generate (1-4096)
- **temperature**: Sampling temperature (0.1-2.0)
  - Lower = more focused and deterministic
  - Higher = more creative and diverse
- **top_p**: Nucleus sampling (0.1-0.99)
  - Lower = more focused on likely tokens
- **top_k**: Top-k sampling (1-100)
  - Lower = more focused token selection

### Model Configuration

The server automatically loads Qwen3-0.6B with these specs:
- 28 transformer layers
- 1024 hidden dimensions
- 16 attention heads (GQA with 8 KV heads)
- 151,936 vocabulary size
- 40,960 max sequence length

## Error Handling

The server includes comprehensive error handling:

- **Model Loading Errors**: Automatic fallback and clear error messages
- **Invalid Requests**: Pydantic validation with detailed error responses
- **Generation Errors**: Graceful handling with proper HTTP status codes
- **Device Mismatches**: Automatic tensor device management

## Performance Tips

- **GPU Usage**: Always prefer GPU for faster inference
- **Batch Size**: Current implementation handles single requests (batching can be added later)
- **Memory**: Qwen3-0.6B requires ~2-4GB GPU RAM
- **Caching**: Models are cached locally after first download
