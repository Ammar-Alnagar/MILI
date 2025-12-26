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

# Import MILI components
import sys
from pathlib import Path

# Add project to path (same as test_real_weights.py)
project_path = str(Path(__file__).parent)
sys.path.insert(0, project_path)
print(f"DEBUG: Added to path: {project_path}")
print(f"DEBUG: sys.path[0]: {sys.path[0]}")

from python_layer.model.weight_loader import WeightLoader
from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode
from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer


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
model: Optional[Qwen3Model] = None
tokenizer: Optional[QwenTokenizer] = None


@app.on_event("startup")
async def startup():
    """Initialize model and tokenizer on startup."""
    global model, tokenizer

    try:
        print("🚀 Initializing MILI Qwen3 Inference Server...")

        # Initialize tokenizer
        tokenizer = QwenTokenizer()
        print("✅ Tokenizer initialized")

        # Configure model for Qwen3-0.6B
        config = ModelConfig(
            vocab_size=151936,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            num_layers=2,  # Small for demo, can increase
            max_seq_length=512,
            dtype="float32",
        )
        print("✅ Model config created")

        # Try to load real weights
        try:
            model_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
            )
            weight_loader = WeightLoader(
                model_path=model_path,
                config={
                    "vocab_size": 151936,
                    "hidden_size": 1024,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 28,
                },
                device="cpu",
                dtype="float32",
            )
            weight_loader.load_from_safetensors()
            print("✅ Loaded real Qwen3-0.6B weights")
        except Exception as e:
            print(f"⚠️  Weight loading failed: {e}, using random weights")
            weight_loader = None

        # Initialize model
        model = Qwen3Model(config, weight_loader)
        print("✅ MILI Qwen3 model loaded and ready for inference")
        print(
            f"📊 Model: {config.num_layers} layers, {config.hidden_size} hidden, {config.vocab_size} vocab"
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
        prompt_tokens = tokenizer.encode(request.prompt)
        print(f"📝 Prompt: '{request.prompt}' ({len(prompt_tokens)} tokens)")

        # Generate tokens
        generated_tokens = model.generate(
            input_ids=np.array(prompt_tokens),
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Decode response
        full_tokens = generated_tokens.tolist()
        generated_text = tokenizer.decode(full_tokens[len(prompt_tokens) :])

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
        "model_type": "Qwen3-0.6B (MILI)",
        "vocab_size": 151936,
        "max_seq_length": 512,
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
