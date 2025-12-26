# Python Integration Guide for MILI

## Introduction

This guide covers building the Python layer that interfaces with Mojo GPU kernels, manages model weights, handles tokenization, and orchestrates the inference pipeline.

### Layer Architecture

```
┌─────────────────────────────────┐
│      Python API (FastAPI)       │  <- REST endpoints
└────────────────────┬────────────┘
                     │
┌────────────────────▼────────────┐
│   Request Scheduler (Python)    │  <- Continuous batching
└────────────────────┬────────────┘
                     │
┌────────────────────▼────────────┐
│    Model & Weight Manager       │  <- Model loading
└────────────────────┬────────────┘
                     │
┌────────────────────▼────────────┐
│      Mojo Kernel Bindings       │  <- C FFI to Mojo
└────────────────────┬────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────▼──┐            ┌─────┬─┘
    │ Attention Kernels  │ Other Kernels
    └────────┘           └──────────
```

---

## Part 1: Model Architecture

### 1.1 Qwen3 Configuration

**File: `python_layer/model/config.py`**

```python
"""Qwen3 model configuration."""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""
    
    # Model architecture
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For Grouped Query Attention
    vocab_size: int = 150000
    max_position_embeddings: int = 32768
    
    # Activation functions
    hidden_act: str = "silu"  # SwiGLU
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_dropout: float = 0.0
    
    # Quantization (optional)
    quantization_config: Optional[dict] = None
    
    # Rope parameters
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "head_dim": self.head_dim,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Qwen3Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen3Config":
        """Load config from HuggingFace."""
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name)
            return cls.from_dict(hf_config.to_dict())
        except Exception as e:
            print(f"Error loading config from {model_name}: {e}")
            return cls()


# Preset configurations
QWEN3_0.6B = Qwen3Config(
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
)

QWEN3_1.7B = Qwen3Config(
    hidden_size=5120,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=10,
)

QWEN3_4B = Qwen3Config(
    hidden_size=8192,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
)
```

### 1.2 Weight Loading

**File: `python_layer/model/weight_loader.py`**

```python
"""Weight loading utilities for model initialization."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import safetensors.torch
from .config import Qwen3Config


class WeightLoader:
    """Load and manage model weights."""
    
    def __init__(
        self,
        model_path: str,
        config: Qwen3Config,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.device = device
        self.dtype = dtype
        self.weights: Dict[str, torch.Tensor] = {}
    
    def load_from_safetensors(self) -> Dict[str, torch.Tensor]:
        """Load weights from safetensors format."""
        safetensors_path = self.model_path / "model.safetensors"
        
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
        
        self.weights = safetensors.torch.load_file(
            str(safetensors_path),
            device=self.device
        )
        return self.weights
    
    def load_from_huggingface(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Load weights from HuggingFace model hub."""
        try:
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            
            self.weights = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }
            return self.weights
        except Exception as e:
            raise RuntimeError(f"Error loading from HuggingFace: {e}")
    
    def quantize_to_fp8(self) -> Dict[str, torch.Tensor]:
        """Quantize weights to FP8 for memory efficiency."""
        quantized = {}
        
        for name, weight in self.weights.items():
            # Skip normalization and embedding layers
            if "norm" in name or "embed" in name:
                quantized[name] = weight
                continue
            
            # Quantize to FP8
            max_val = weight.abs().max()
            scale = max_val / 127.0
            quantized_weight = (weight / scale).to(torch.int8).float() * scale
            quantized[name] = quantized_weight
        
        self.weights = quantized
        return self.weights
    
    def get_weight(self, name: str) -> torch.Tensor:
        """Get a specific weight tensor."""
        if name not in self.weights:
            raise KeyError(f"Weight {name} not found")
        return self.weights[name]
    
    def get_weight_as_numpy(self, name: str) -> np.ndarray:
        """Get weight as NumPy array."""
        tensor = self.get_weight(name)
        return tensor.cpu().numpy()


class KVCacheAllocator:
    """Allocate and manage KV cache memory."""
    
    def __init__(
        self,
        config: Qwen3Config,
        max_batch_size: int = 64,
        max_seq_length: int = 32768,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.config = config
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.device = device
        self.page_size = 16  # Tokens per page
        
        # Allocate cache memory
        self._allocate_cache()
    
    def _allocate_cache(self):
        """Pre-allocate KV cache."""
        num_pages = (self.max_seq_length // self.page_size) + 1
        
        # K cache: [num_pages, page_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            (num_pages, self.page_size, 
             self.config.num_key_value_heads, 
             self.config.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # V cache: same shape
        self.v_cache = torch.zeros_like(self.k_cache)
        
        # Track which pages are used
        self.page_usage = [False] * num_pages
        self.page_ref_count = [0] * num_pages
    
    def allocate_page(self) -> int:
        """Allocate a free page, return page index."""
        for i, used in enumerate(self.page_usage):
            if not used:
                self.page_usage[i] = True
                self.page_ref_count[i] = 1
                return i
        raise RuntimeError("No free pages in KV cache")
    
    def free_page(self, page_id: int):
        """Free a page."""
        if self.page_ref_count[page_id] > 0:
            self.page_ref_count[page_id] -= 1
            if self.page_ref_count[page_id] == 0:
                self.page_usage[page_id] = False
    
    def add_page_reference(self, page_id: int):
        """Add reference to page (for sharing)."""
        self.page_ref_count[page_id] += 1
```

---

## Part 2: Tokenization

### 2.1 Tokenizer Wrapper

**File: `python_layer/tokenizer/qwen_tokenizer.py`**

```python
"""Qwen tokenizer wrapper using tiktoken."""

import tiktoken
from typing import List, Union
import numpy as np


class QwenTokenizer:
    """Tokenizer for Qwen models."""
    
    def __init__(self, model_name: str = "qwen3"):
        """Initialize tokenizer."""
        try:
            # Try to load Qwen-specific encoding
            self.encoding = tiktoken.get_encoding(model_name)
        except KeyError:
            # Fall back to cl100k_base
            print(f"Warning: {model_name} encoding not found, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.vocab_size = len(self.encoding)
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: String or list of strings to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Token IDs or list of token ID lists
        """
        if isinstance(text, str):
            return self.encoding.encode(text)
        else:
            return [self.encoding.encode(t) for t in text]
    
    def decode(
        self,
        tokens: Union[List[int], np.ndarray],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        return self.encoding.decode(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 2048,
        padding: bool = True,
        return_tensors: bool = False
    ) -> dict:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            return_tensors: Whether to return as torch tensors
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        input_ids = []
        attention_mask = []
        
        for text in texts:
            tokens = self.encode(text)
            
            # Truncate if necessary
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Pad if necessary
            if padding and len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens = tokens + [0] * pad_length
                mask = [1] * (max_length - pad_length) + [0] * pad_length
            else:
                mask = [1] * len(tokens)
            
            input_ids.append(tokens)
            attention_mask.append(mask)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if return_tensors:
            import torch
            result["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        
        return result


# Singleton tokenizer instance
_tokenizer = None

def get_tokenizer() -> QwenTokenizer:
    """Get or create singleton tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = QwenTokenizer()
    return _tokenizer
```

---

## Part 3: Inference Scheduler

### 3.1 Request Scheduler with Continuous Batching

**File: `python_layer/inference/scheduler.py`**

```python
"""Request scheduler with continuous batching."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from queue import Queue
from enum import Enum
import time
import uuid


class RequestPhase(Enum):
    """Inference phase for a request."""
    PREFILL = 1  # Processing initial prompt
    DECODE = 2   # Generating tokens one by one


@dataclass
class RequestMetadata:
    """Metadata for an inference request."""
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    
    # Runtime state
    phase: RequestPhase = RequestPhase.PREFILL
    generated_tokens: List[int] = field(default_factory=list)
    kv_cache_pages: List[int] = field(default_factory=list)
    seq_length: int = 0
    created_at: float = field(default_factory=time.time)
    
    def is_complete(self) -> bool:
        """Check if request is complete."""
        return len(self.generated_tokens) >= self.max_tokens
    
    def tokens_remaining(self) -> int:
        """Get number of tokens remaining to generate."""
        return self.max_tokens - len(self.generated_tokens)


class ContinuousBatchScheduler:
    """
    Scheduler that maintains a continuous batch of requests.
    
    Key features:
    - Dynamically add/remove requests
    - Interleave prefill and decode phases
    - Maximize GPU utilization
    """
    
    def __init__(
        self,
        max_batch_size: int = 64,
        prefill_batch_size: int = 32,
        decode_batch_size: int = 256,
        max_seq_length: int = 32768
    ):
        self.max_batch_size = max_batch_size
        self.prefill_batch_size = prefill_batch_size
        self.decode_batch_size = decode_batch_size
        self.max_seq_length = max_seq_length
        
        # Request queues
        self.pending_requests: Queue[RequestMetadata] = Queue()
        self.active_requests: Dict[str, RequestMetadata] = {}
        self.completed_requests: List[RequestMetadata] = []
    
    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Add a new inference request.
        
        Args:
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        request = RequestMetadata(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.pending_requests.put(request)
        return request_id
    
    def get_next_batch(self) -> Tuple[List[RequestMetadata], str]:
        """
        Get next batch of requests to process.
        
        Returns:
            (batch of requests, batch_type: 'prefill' or 'decode')
        """
        batch = []
        batch_type = "prefill" if self.pending_requests.qsize() > 0 else "decode"
        
        if batch_type == "prefill":
            # Prefill phase: prioritize new requests
            max_size = min(self.prefill_batch_size, self.max_batch_size)
            
            while len(batch) < max_size and not self.pending_requests.empty():
                request = self.pending_requests.get()
                
                # Initialize runtime state
                request.phase = RequestPhase.PREFILL
                request.seq_length = len(request.prompt_tokens)
                
                batch.append(request)
                self.active_requests[request.request_id] = request
        
        else:
            # Decode phase: process active requests
            max_size = min(self.decode_batch_size, self.max_batch_size)
            
            for req_id, request in list(self.active_requests.items()):
                if len(batch) >= max_size:
                    break
                
                if not request.is_complete():
                    request.phase = RequestPhase.DECODE
                    batch.append(request)
                else:
                    # Request is complete, move to completed
                    self.completed_requests.append(request)
                    del self.active_requests[req_id]
        
        return batch, batch_type
    
    def update_request(
        self,
        request_id: str,
        new_token: int,
        kv_cache_page: Optional[int] = None
    ):
        """
        Update request with generated token.
        
        Args:
            request_id: ID of request to update
            new_token: Newly generated token ID
            kv_cache_page: Page ID allocated for new tokens
        """
        if request_id not in self.active_requests:
            return
        
        request = self.active_requests[request_id]
        request.generated_tokens.append(new_token)
        request.seq_length += 1
        
        if kv_cache_page is not None:
            request.kv_cache_pages.append(kv_cache_page)
    
    def get_request_status(self, request_id: str) -> dict:
        """Get status of a request."""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            return {
                "status": "processing",
                "phase": req.phase.name,
                "tokens_generated": len(req.generated_tokens),
                "tokens_remaining": req.tokens_remaining(),
            }
        
        for req in self.completed_requests:
            if req.request_id == request_id:
                return {
                    "status": "completed",
                    "tokens_generated": len(req.generated_tokens),
                }
        
        return {"status": "not_found"}
    
    def get_completed_result(self, request_id: str) -> Optional[Dict]:
        """Get result of a completed request."""
        for req in self.completed_requests:
            if req.request_id == request_id:
                return {
                    "request_id": request_id,
                    "generated_tokens": req.generated_tokens,
                    "total_tokens": len(req.prompt_tokens) + len(req.generated_tokens),
                    "generation_time": time.time() - req.created_at,
                }
        return None
```

### 3.2 Sampling Strategies

**File: `python_layer/inference/sampler.py`**

```python
"""Sampling strategies for token generation."""

import numpy as np
import torch
from typing import Tuple


class Sampler:
    """Token sampling from logits."""
    
    @staticmethod
    def top_k_top_p_sampling(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Apply top-k and top-p (nucleus) sampling.
        
        Args:
            logits: Model logits [batch_size, vocab_size]
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k most likely tokens
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Sampled token IDs
        """
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # Keep at least one token
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample from filtered distribution
        probs = torch.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens
    
    @staticmethod
    def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
        """Select token with highest probability."""
        return torch.argmax(logits, dim=-1)
    
    @staticmethod
    def temperature_sampling(
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample with temperature scaling."""
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

---

## Part 4: Model Class

**File: `python_layer/model/qwen3_model.py`**

```python
"""Qwen3 model wrapper."""

import torch
from typing import List, Dict, Optional
from .config import Qwen3Config
from .weight_loader import WeightLoader, KVCacheAllocator
from ..tokenizer.qwen_tokenizer import QwenTokenizer


class Qwen3Model:
    """Qwen3 language model for inference."""
    
    def __init__(
        self,
        config: Qwen3Config,
        weight_loader: WeightLoader,
        device: str = "cuda"
    ):
        self.config = config
        self.weight_loader = weight_loader
        self.device = device
        
        # Initialize cache allocator
        self.cache_allocator = KVCacheAllocator(
            config,
            max_batch_size=64,
            device=device
        )
        
        # Initialize tokenizer
        self.tokenizer = QwenTokenizer()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            kv_cache: Previous KV cache for incremental generation
            
        Returns:
            (logits [batch_size, seq_length, vocab_size], new_kv_cache)
        """
        batch_size, seq_length = input_ids.shape
        
        # This would call Mojo kernels in actual implementation
        # For now, we show the interface
        
        # TODO: Call Mojo embedding kernel
        # TODO: Call Mojo transformer layers
        # TODO: Return logits and KV cache
        
        pass
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        generated_tokens = []
        kv_cache = None
        
        for _ in range(max_tokens):
            # Forward pass
            logits, kv_cache = self.forward(
                input_ids[:, -1:],  # Only last token for decode
                kv_cache=kv_cache
            )
            
            # Sample next token
            # TODO: Implement sampling
            
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode to text
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text
```

---

## Part 5: Setup and Installation

### 5.1 Requirements

**File: `requirements.txt`**

```
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
tiktoken>=0.5.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### 5.2 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Mojo (if not already installed)
curl https://docs.modular.com/mojo/install.sh | bash
```

---

## Next Steps

1. Implement the model forward pass with Mojo kernel calls
2. Build the FastAPI server (see deployment docs)
3. Create integration tests
4. Profile and optimize performance

---

## Testing Python Components

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run specific test
pytest tests/unit/test_tokenizer.py::test_encode -v
```
