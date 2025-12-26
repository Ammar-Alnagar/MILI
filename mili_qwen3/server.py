#!/usr/bin/env python3
"""
MILI Qwen3 Inference Server
Standalone server script to serve the Qwen3 model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import torch

# Import transformers for the actual Qwen3 model
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available")
    exit(1)


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
        print("🚀 Initializing Qwen3 Inference Server...")

        # Load model and tokenizer from HuggingFace
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model from: {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Tokenizer loaded")

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        print("✅ Model loaded")

        print(
            f"📊 Model: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden, {model.config.vocab_size} vocab"
        )

    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
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
        print(f"📝 Prompt: '{request.prompt}' ({len(prompt_tokens)} tokens)")

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
        generated_text = tokenizer.decode(
            full_tokens[len(prompt_tokens) :], skip_special_tokens=True
        )

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
        print(f"❌ Generation error: {e}")
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

    print("🎯 Starting MILI Qwen3 Inference Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
