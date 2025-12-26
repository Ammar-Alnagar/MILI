# MILI Implementation Guide

## Overview

MILI (Machine Learning Inference Lattice for Qwen3) is a production-grade inference framework for the Qwen3 language model with Mojo GPU kernel acceleration and advanced memory management.

## Architecture

### 1. Core Components

#### Mojo Kernels (`mojo_kernels/`)
- **Activations** (`core/activations.🔥`): SwiGLU, GELU, ReLU, SiLU
- **Attention** (`core/attention.🔥`): FlashAttention, Decode Attention, GQA
- **Normalization** (`core/normalization.🔥`): RMSNorm with residual fusion
- **RoPE** (`core/rope.🔥`): Rotary Position Embeddings with frequency precomputation
- **KV Cache** (`memory/kv_cache.🔥`): Paged KV cache with RadixAttention
- **Types** (`utils/types.🔥`): Core type definitions and GPU configuration

#### Python Layer (`python_layer/`)
- **Model** (`model/qwen_model.py`): Main Qwen3 model implementation
- **Memory** (`memory/kv_cache_manager.py`): Python KV cache management
- **Inference** (`inference/inference_engine.py`): Inference engine with continuous batching
- **Tokenizer** (`tokenizer/qwen_tokenizer.py`): BPE tokenizer and chat formatting

### 2. Key Features

#### Paged KV Cache
- Fixed-size pages (typically 16 tokens/page)
- Reference counting for multi-request sharing
- Independent block allocation/deallocation
- Efficient gather/scatter operations

#### RadixAttention with Prefix Sharing
- Radix tree structure for prompt prefixes
- Multiple requests share common prefixes
- Reduces redundant computation and memory

#### Continuous Batching
- Dynamic request scheduling
- Token-based batching for efficiency
- Support for variable-length sequences

#### Flash Attention
- Efficient attention computation
- Block-wise tiling to reduce memory I/O
- Online softmax for numerical stability

## Implementation Details

### Mojo Kernel Implementation Pattern

Each kernel follows a consistent pattern:

```mojo
struct MyKernel:
    var context: DeviceContext
    var metadata: TensorMetadata
    var config: MyConfig
    
    fn __init__(context, metadata, config) -> Self:
        return Self(context=context, metadata=metadata, config=config)
    
    fn forward(inout self, input_ptr, output_ptr):
        # Implementation here
        self.kernel_launch(input_ptr, output_ptr)
    
    fn kernel_launch(inout self, input_ptr, output_ptr):
        # CUDA-style grid/block execution
        let total_elements = ...
        let block_size = self.config.threads_per_block
        let grid_size = (total_elements + block_size - 1) // block_size
        
        for block_idx in range(grid_size):
            for thread_idx in range(block_size):
                self.compute_thread(thread_idx, block_idx, input_ptr, output_ptr)
```

### KV Cache Management Flow

```
Request -> Allocate Pages -> Forward Pass -> Cache KV -> Next Request
                    ↓                            ↓
           RadixAttention                   Update Block
           Prefix Sharing                   Token Count
                    ↓                            ↓
           Request Complete -> Free Pages -> Eviction Check
```

### Attention Computation

#### Prefill Phase
```
Input: Q[batch, seq_len, heads, dim]
       K[batch, seq_len, heads, dim]
       V[batch, seq_len, heads, dim]

1. Apply RoPE to Q and K
2. Compute scores: Q @ K^T / sqrt(d_k)
3. Apply causal mask
4. Softmax with online algorithm
5. Apply to values: scores @ V
6. Merge heads
7. Output projection
```

#### Decode Phase
```
Input: Q_new[1, heads, dim]
       K_cache[cache_len, heads, dim]
       V_cache[cache_len, heads, dim]

1. Apply RoPE to Q_new
2. Compute scores: Q_new @ K_cache^T / sqrt(d_k)
3. Softmax
4. Apply to cached values
5. Output projection
```

## Python Layer Usage

### Basic Initialization

```python
from python_layer import (
    Qwen3Model, ModelConfig, PagedKVCache, 
    QwenTokenizer, InferenceEngine
)

# Create configuration
config = ModelConfig(
    vocab_size=151936,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    num_layers=32
)

# Initialize components
model = Qwen3Model(config)
tokenizer = QwenTokenizer()
cache = PagedKVCache(page_size=16, num_pages=1024)
engine = InferenceEngine(model, cache, batch_size=32)
```

### Text Generation

```python
import numpy as np

# Tokenize prompt
prompt = "What is machine learning?"
tokens = tokenizer.encode(prompt)
input_ids = np.array(tokens, dtype=np.int32)

# Generate
generated_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Decode
output_text = tokenizer.decode(generated_ids)
print(output_text)
```

### Chat Interface

```python
from python_layer import MessageFormatter

formatter = MessageFormatter(tokenizer)

# Format conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How does attention work?"}
]

# Encode
tokens = formatter.encode_chat(messages)
generated = model.generate(tokens, max_new_tokens=200)

# Parse response
response_text = formatter.parse_response(
    tokenizer.decode(generated)
)
print(response_text)
```

### Batch Processing

```python
from python_layer import InferenceRequest

# Add multiple requests
for i in range(10):
    request = InferenceRequest(
        request_id=i,
        input_ids=[1, 2, 3],
        max_new_tokens=50
    )
    engine.add_request(request)

# Process in batches
while engine.active_requests or engine.pending_requests:
    completed = engine.step()
    for req_id, output in completed:
        print(f"Request {req_id}: {output.token_ids}")
```

## Performance Considerations

### Memory Optimization
1. **Paging**: Use page size matching GPU cache line (typically 128 bytes)
2. **Prefix Sharing**: Enable for similar prompts (saves 20-40% memory)
3. **Quantization**: Support for int8/fp8 (future enhancement)

### Computational Efficiency
1. **Flash Attention**: Reduces memory I/O by 10x
2. **Kernel Fusion**: Fuse operations where possible
3. **Batch Size**: Tune based on memory constraints

### Latency Targets
- Prefill: 20-40 µs per token
- Decode: 100-200 µs per token
- Memory BW: 2 TB/s on H100

## Testing

### Unit Tests
```bash
python -m pytest tests/unit/test_tokenizer.py
```

### Integration Tests
```bash
python -m pytest tests/integration/test_inference.py
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## Deployment

### Requirements
- CUDA 11.8+
- Mojo 0.1.0+
- Python 3.8+
- numpy, torch (optional)

### Installation
```bash
pip install -r requirements.txt
cd mojo_kernels && bash build.sh
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
RUN apt-get install -y mojo python3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce batch size
- Enable prefix sharing
- Use smaller page size

#### Slow Generation
- Check GPU utilization
- Verify batch size configuration
- Profile with nvprof

#### Cache Corruption
- Enable cache validation in debug mode
- Check page allocation/deallocation
- Verify KV write correctness

## Future Enhancements

1. **Quantization**: int8/fp8 KV cache
2. **Multi-GPU**: Tensor parallelism
3. **Speculative Decoding**: Draft models
4. **Tree Attention**: For tree-based generation
5. **Custom Kernels**: User-defined operations

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- Shazeer et al., "Fast Transformer Decoding: One Write-Head is All You Need", 2019
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023
