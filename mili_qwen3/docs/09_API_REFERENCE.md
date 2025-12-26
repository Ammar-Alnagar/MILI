# MILI API Reference Documentation

## Overview

Complete API reference for all public classes, methods, and functions in the MILI system.

---

## Python Layer API

### Model Configuration

#### `Qwen3Config`

```python
class Qwen3Config:
    """Configuration for Qwen3 model."""
    
    # Model Architecture
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 150000
    max_position_embeddings: int = 32768
    
    # Activation & Normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_dropout: float = 0.0
    
    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Properties
    head_dim: int  # Read-only: hidden_size // num_attention_heads
```

**Methods**:

```python
def __post_init__(self):
    """Validate and initialize configuration."""
    
def to_dict(self) -> dict:
    """Convert config to dictionary."""
    
@classmethod
def from_dict(cls, config_dict: dict) -> "Qwen3Config":
    """Create config from dictionary."""
    
@classmethod
def from_pretrained(cls, model_name: str) -> "Qwen3Config":
    """Load config from HuggingFace."""
```

**Example**:

```python
from python_layer.model.config import Qwen3Config

# Create config
config = Qwen3Config(
    hidden_size=4096,
    num_hidden_layers=32
)

# Load from HuggingFace
config = Qwen3Config.from_pretrained("Qwen/Qwen3-7B")

# Convert to dict
config_dict = config.to_dict()
```

---

### Weight Management

#### `WeightLoader`

```python
class WeightLoader:
    """Load and manage model weights."""
    
    def __init__(
        self,
        model_path: str,
        config: Qwen3Config,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize weight loader."""
```

**Methods**:

```python
def load_from_safetensors(self) -> Dict[str, torch.Tensor]:
    """Load weights from safetensors format."""
    
def load_from_huggingface(self, model_name: str) -> Dict[str, torch.Tensor]:
    """Load weights from HuggingFace model hub."""
    
def quantize_to_fp8(self) -> Dict[str, torch.Tensor]:
    """Quantize weights to FP8 for memory efficiency."""
    
def get_weight(self, name: str) -> torch.Tensor:
    """Get a specific weight tensor."""
    
def get_weight_as_numpy(self, name: str) -> np.ndarray:
    """Get weight as NumPy array."""
```

**Example**:

```python
from python_layer.model.weight_loader import WeightLoader

loader = WeightLoader("./models/Qwen3-7B", config)
weights = loader.load_from_huggingface("Qwen/Qwen3-7B")
quantized = loader.quantize_to_fp8()
```

---

### KV Cache Management

#### `KVCacheAllocator`

```python
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
        """Initialize KV cache allocator."""
```

**Methods**:

```python
def allocate_page(self) -> int:
    """Allocate a free page, return page index."""
    # Returns: page_id (int) or raises RuntimeError if exhausted
    
def free_page(self, page_id: int):
    """Free a page."""
    
def add_page_reference(self, page_id: int):
    """Add reference to page (for sharing)."""
    
def get_page(self, page_id: int, cache_type: str = 'k') -> torch.Tensor:
    """Get a single page."""
    # cache_type: 'k' or 'v'
    # Returns: [page_size, num_heads, head_dim]
    
def gather_pages(
    self,
    page_ids: List[int],
    seq_len: int,
    cache_type: str = 'k'
) -> torch.Tensor:
    """Gather pages into contiguous tensor."""
    # Returns: [seq_len, num_heads, head_dim]
    
def write_pages(
    self,
    page_ids: List[int],
    data: torch.Tensor,
    cache_type: str = 'k'
):
    """Write data to pages."""
```

**Example**:

```python
from python_layer.model.weight_loader import KVCacheAllocator

allocator = KVCacheAllocator(config, max_batch_size=64)

# Allocate pages
page_ids = allocator.allocate_page()  # Returns single page_id

# Write data
allocator.write_pages([page_ids], k_tensor, cache_type='k')

# Read data
k_cache = allocator.gather_pages([page_ids], seq_len=16, cache_type='k')
```

---

### Tokenization

#### `QwenTokenizer`

```python
class QwenTokenizer:
    """Tokenizer for Qwen models."""
    
    def __init__(self, model_name: str = "qwen3"):
        """Initialize tokenizer."""
```

**Methods**:

```python
def encode(
    self,
    text: Union[str, List[str]],
    add_special_tokens: bool = True
) -> Union[List[int], List[List[int]]]:
    """Encode text to token IDs."""
    
def decode(
    self,
    tokens: Union[List[int], np.ndarray],
    skip_special_tokens: bool = True
) -> str:
    """Decode token IDs back to text."""
    
def batch_encode(
    self,
    texts: List[str],
    max_length: int = 2048,
    padding: bool = True,
    return_tensors: bool = False
) -> dict:
    """Batch encode multiple texts."""
    # Returns: {"input_ids": [...], "attention_mask": [...]}
```

**Example**:

```python
from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer, get_tokenizer

tokenizer = QwenTokenizer()

# Single text
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Batch
batch = tokenizer.batch_encode(
    ["Hello", "World"],
    max_length=512,
    return_tensors=True
)

# Get singleton
tokenizer = get_tokenizer()
```

---

### Request Scheduling

#### `ContinuousBatchScheduler`

```python
class ContinuousBatchScheduler:
    """Scheduler with continuous batching."""
    
    def __init__(
        self,
        max_batch_size: int = 64,
        prefill_batch_size: int = 32,
        decode_batch_size: int = 256,
        max_seq_length: int = 32768
    ):
        """Initialize scheduler."""
```

**Methods**:

```python
def add_request(
    self,
    prompt_tokens: List[int],
    max_tokens: int = 128,
    **kwargs
) -> str:
    """
    Add a new inference request.
    
    Returns: request_id (str)
    """
    
def get_next_batch(self) -> Tuple[List[RequestMetadata], str]:
    """
    Get next batch of requests.
    
    Returns: (batch_list, batch_type: 'prefill' or 'decode')
    """
    
def update_request(
    self,
    request_id: str,
    new_token: int,
    kv_cache_page: Optional[int] = None
):
    """Update request with generated token."""
    
def get_request_status(self, request_id: str) -> dict:
    """Get status of a request."""
    # Returns: {"status": "processing"|"completed"|"not_found", ...}
    
def get_completed_result(self, request_id: str) -> Optional[Dict]:
    """Get result of a completed request."""
```

**Example**:

```python
from python_layer.inference.scheduler import ContinuousBatchScheduler

scheduler = ContinuousBatchScheduler(max_batch_size=64)

# Add request
req_id = scheduler.add_request(
    prompt_tokens=[1, 2, 3],
    max_tokens=100,
    temperature=0.7
)

# Get batch
batch, batch_type = scheduler.get_next_batch()

# Update with generated token
scheduler.update_request(req_id, new_token=5)

# Check status
status = scheduler.get_request_status(req_id)
```

---

### Sampling

#### `Sampler`

```python
class Sampler:
    """Token sampling from logits."""
```

**Static Methods**:

```python
@staticmethod
def top_k_top_p_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> torch.Tensor:
    """Apply top-k and top-p sampling."""
    # Returns: sampled_token_ids [batch_size]
    
@staticmethod
def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Select token with highest probability."""
    
@staticmethod
def temperature_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Sample with temperature scaling."""
```

**Example**:

```python
from python_layer.inference.sampler import Sampler

logits = torch.randn(1, 150000)

# Top-k + Top-p
tokens = Sampler.top_k_top_p_sampling(
    logits,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Greedy
tokens = Sampler.greedy_sampling(logits)
```

---

### Model Class

#### `Qwen3Model`

```python
class Qwen3Model:
    """Qwen3 language model for inference."""
    
    def __init__(
        self,
        config: Qwen3Config,
        weight_loader: WeightLoader,
        device: str = "cuda"
    ):
        """Initialize model."""
```

**Methods**:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[Dict] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Forward pass through model.
    
    Args:
        input_ids: [batch_size, seq_length]
        attention_mask: [batch_size, seq_length]
        kv_cache: Previous KV cache for incremental generation
        
    Returns:
        (logits [batch_size, seq_length, vocab_size], new_kv_cache)
    """
    
def generate(
    self,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50
) -> str:
    """Generate text from prompt."""
```

**Example**:

```python
from python_layer.model.qwen3_model import Qwen3Model

model = Qwen3Model(config, weight_loader)

# Generate
output = model.generate(
    "What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

print(output)
```

---

## FastAPI Server API

### Endpoints

#### `POST /generate`

Generate text from a prompt.

**Request**:
```json
{
    "prompt": "What is AI?",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "stream": false
}
```

**Response**:
```json
{
    "request_id": "uuid-string",
    "generated_text": "AI stands for...",
    "tokens": [123, 456, 789],
    "total_tokens": 131
}
```

**Python Client**:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Hello",
        "max_tokens": 50
    }
)
result = response.json()
print(result["generated_text"])
```

#### `GET /status/{request_id}`

Get status of a generation request.

**Response**:
```json
{
    "status": "processing|completed|not_found",
    "tokens_generated": 50,
    "tokens_remaining": 28
}
```

#### `GET /health`

Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "scheduler_ready": true
}
```

---

## Mojo Kernels API

### RoPE Kernel

```mojo
struct RoPEKernel:
    """Kernel for Rotary Position Embeddings."""
    
    fn compute_rope_freqs(
        self,
        pos: UInt32,
        head_dim: UInt32,
        base: Float32 = 10000.0
    ) -> SIMD[DType.float32, 2]:
        """Compute RoPE frequencies for a position."""
        
    fn apply_rope(
        inout self,
        q: SIMD[DType.float32, 2],
        freqs: SIMD[DType.float32, 2],
        dim_idx: Int
    ) -> SIMD[DType.float32, 2]:
        """Apply rope rotation to query/key pair."""
        
    fn forward(
        inout self,
        q_ptr: UnsafePointer[Float32],
        k_ptr: UnsafePointer[Float32],
        seq_length: UInt32
    ):
        """Apply RoPE to embeddings."""
```

---

### RMSNorm Kernel

```mojo
struct RMSNormKernel:
    """Kernel for RMSNorm normalization."""
    
    fn compute_rms(
        self,
        ptr: UnsafePointer[Float32],
        dim: UInt32
    ) -> Float32:
        """Compute RMS (root mean square)."""
        
    fn forward(
        inout self,
        input_ptr: UnsafePointer[Float32],
        weight_ptr: UnsafePointer[Float32],
        output_ptr: UnsafePointer[Float32],
        hidden_dim: UInt32
    ):
        """Apply RMSNorm normalization."""
```

---

### Attention Kernels

```mojo
struct FlashAttentionKernel:
    """High-performance FlashAttention kernel."""
    
    fn attention_forward(
        inout self,
        q: UnsafePointer[Float32],
        k: UnsafePointer[Float32],
        v: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        seq_length: UInt32
    ):
        """Compute FlashAttention forward pass."""

struct DecodeAttentionKernel:
    """Optimized decode-phase attention."""
    
    fn forward(
        inout self,
        q: UnsafePointer[Float32],
        k_cache: UnsafePointer[Float32],
        v_cache: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        seq_length: UInt32
    ):
        """Single-token decode attention."""
```

---

## Paged KV Cache API

### `PagedKVCache`

```python
class PagedKVCache:
    """Paged KV cache with reference counting."""
    
    def allocate_pages(self, num_pages: int) -> Optional[List[int]]:
        """Allocate contiguous or non-contiguous pages."""
        # Returns: List of page IDs
        
    def add_reference(self, page_ids: List[int]):
        """Add reference to pages (for sharing)."""
        
    def free_pages(self, page_ids: List[int]):
        """Free pages by decrementing reference count."""
        
    def get_page(self, page_id: int, cache_type: str = 'k') -> torch.Tensor:
        """Get a single page."""
        
    def gather_pages(
        self,
        page_ids: List[int],
        seq_len: int,
        cache_type: str = 'k'
    ) -> torch.Tensor:
        """Gather pages into contiguous tensor."""
```

---

### `RadixAttentionCache`

```python
class RadixAttentionCache:
    """Cache manager with RadixAttention for prefix sharing."""
    
    def add_request(
        self,
        request_id: str,
        prompt_tokens: List[int]
    ) -> Dict:
        """Add request, potentially sharing prefix."""
        # Returns: metadata with page allocation
        
    def remove_request(self, request_id: str):
        """Remove request and free pages."""
        
    def get_kv_cache(
        self,
        request_id: str,
        cache_type: str = 'k'
    ) -> torch.Tensor:
        """Get full KV cache for request."""
```

---

## Configuration Files API

### Model Config (JSON)

```json
{
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 150000,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0
}
```

### Inference Config (JSON)

```json
{
    "inference": {
        "device": "cuda",
        "dtype": "float16",
        "max_batch_size": 64
    },
    "kv_cache": {
        "page_size": 16,
        "num_pages": 1024
    },
    "sampling": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50
    }
}
```

---

## Type Definitions

### Python Types

```python
from dataclasses import dataclass
from enum import Enum

class RequestPhase(Enum):
    PREFILL = 1
    DECODE = 2

@dataclass
class RequestMetadata:
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int
    generated_tokens: List[int]
    phase: RequestPhase
    seq_length: int
```

---

## Error Handling

### Common Exceptions

```python
# Raised when out of memory
RuntimeError("CUDA out of memory")

# Raised when weights not found
FileNotFoundError("Model files not found")

# Raised when request invalid
ValueError("Invalid request parameters")

# Raised when cache exhausted
RuntimeError("Insufficient pages in KV cache")

# Raised when model not loaded
RuntimeError("Model not loaded")
```

---

## Performance Tips

### API Usage Best Practices

1. **Batch Requests**: Submit multiple requests together for better throughput
2. **Reuse Tokenizer**: Use singleton pattern for tokenizer instance
3. **Pre-allocate Memory**: Use memory pooling for large-scale deployment
4. **Monitor Metrics**: Track GPU utilization and latency
5. **Cache Results**: Cache frequently accessed data

---

## Version Information

- **MILI Version**: 0.1.0
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Mojo**: Latest from Modular
- **CUDA**: 12.0+

---

## See Also

- [Python Integration Guide](03_PYTHON_INTEGRATION.md)
- [Deployment Guide](06_DEPLOYMENT.md)
- [Advanced Optimization](07_ADVANCED_OPTIMIZATION.md)
- [Troubleshooting Guide](08_TROUBLESHOOTING_AND_DEBUGGING.md)
