# MILI: Machine Learning Inference Lattice for Qwen3

## Overview

This comprehensive guide walks you through building a **MILI (Machine Learning Inference Lattice)** implementation optimized for the **Qwen3 architecture**. MILI is a high-performance inference system designed to efficiently serve large language models with minimal latency and maximum throughput.

This project emphasizes **hands-on learning** through manual implementation of optimized kernels in Mojo and Python, enabling you to understand the internals of modern LLM inference systems.

### What You'll Build

By completing this guide, you will have:

1. **High-Performance Mojo Kernels** for GPU-accelerated attention and matrix operations
2. **Python Integration Layer** for model management and request scheduling
3. **KV Cache Management System** with RadixAttention and prefix sharing
4. **Request Scheduler** supporting continuous batching and speculative decoding
5. **Fully Functional Inference Server** capable of serving Qwen3-like models

---

## Target Architecture: Qwen3

### Model Characteristics

- **Decoder-Only Transformer** architecture
- **Grouped Query Attention (GQA)**: Reduces KV cache size while maintaining quality
- **Rotary Position Embeddings (RoPE)**: Efficient positional encoding
- **SwiGLU Activation**: Modern feedforward layer activation
- **RMSNorm Normalization**: Efficient layer normalization
- **Vocabulary**: ~150,000 tokens (using tiktoken)
- **Hidden Dimension**: Configurable (typically 4096-8192)
- **Number of Layers**: Configurable (typically 32-80)
- **Number of Attention Heads**: Configurable with GQA support

### Key Parameters to Configure

```
MODEL CONFIG:
- hidden_size: 4096
- intermediate_size: 11008
- num_attention_heads: 32
- num_key_value_heads: 8  # For GQA
- num_hidden_layers: 32
- vocab_size: 150000
- max_position_embeddings: 32768
```

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     MILI INFERENCE SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FastAPI Server Layer                        │   │
│  │  - Request handling                                      │   │
│  │  - Response formatting                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Python Request Scheduler (Continuous Batching)  │   │
│  │  - Request queue management                             │   │
│  │  - Dynamic batching                                      │   │
│  │  - Prefill/Decode phase management                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Python Model Layer                          │   │
│  │  - Weight loading (safetensors/HuggingFace)             │   │
│  │  - Tokenizer integration (tiktoken)                      │   │
│  │  - Sampling strategies (top-p, top-k, temperature)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         GPU Memory Management                            │   │
│  │  - Paged KV Cache (16-token blocks)                      │   │
│  │  - RadixAttention prefix sharing                         │   │
│  │  - Reference counting for cache blocks                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Mojo GPU Kernels                                 │   │
│  │  ├─ FlashAttention (Prefill)                             │   │
│  │  ├─ Decode-Phase Attention                               │   │
│  │  ├─ RoPE Application                                     │   │
│  │  ├─ RMSNorm                                              │   │
│  │  ├─ SwiGLU Activation                                    │   │
│  │  └─ Optimized GEMM operations                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│              GPU (NVIDIA/AMD with Mojo GPU Support)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Dependencies

### Hardware Requirements

- **GPU**: NVIDIA GPU (Compute Capability 8.0+) or AMD GPU with GCN support
  - Recommended: A100, H100, or RTX 4090
  - Minimum Memory: 24GB VRAM for inference
- **CPU**: Modern multi-core processor (Intel or AMD)
- **RAM**: 32GB+ for model weights and batching

### Software Prerequisites

Before starting, ensure you have:

1. **Mojo SDK**
   ```bash
   # Install from Modular
   curl https://docs.modular.com/mojo/install.sh | bash
   ```

2. **MAX for GPU Support**
   ```bash
   modular install max
   ```

3. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

4. **CUDA Toolkit** (for NVIDIA GPUs)
   ```bash
   nvidia-smi  # Verify installation
   ```

### Python Dependencies

```
torch>=2.0.0          # For reference implementations
transformers>=4.35.0  # For model loading
safetensors>=0.4.0    # For weight loading
tiktoken>=0.5.0       # For tokenization
fastapi>=0.104.0      # For API server
uvicorn>=0.24.0       # For ASGI server
numpy>=1.24.0         # For numerical operations
pytest>=7.4.0         # For testing
```

---

## Project Organization

```
mili_qwen3/
├── docs/
│   ├── 01_PROJECT_OVERVIEW.md          # This file
│   ├── 02_MOJO_KERNEL_GUIDE.md         # Mojo kernel development
│   ├── 03_PYTHON_INTEGRATION.md        # Python layer setup
│   ├── 04_ATTENTION_MECHANISMS.md      # Detailed attention docs
│   ├── 05_KV_CACHE_MANAGEMENT.md       # Cache system docs
│   └── 06_DEPLOYMENT.md                # Deployment guide
│
├── mojo_kernels/
│   ├── core/
│   │   ├── attention.🔥               # FlashAttention kernels
│   │   ├── rope.🔥                    # RoPE kernels
│   │   ├── activations.🔥             # Activation functions
│   │   └── normalization.🔥           # RMSNorm kernels
│   ├── memory/
│   │   ├── kv_cache.🔥                # KV cache management
│   │   └── allocator.🔥               # Memory allocator
│   └── utils/
│       ├── types.🔥                   # Type definitions
│       └── helpers.🔥                 # Utility functions
│
├── python_layer/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── qwen3_model.py              # Model architecture
│   │   ├── weight_loader.py            # Weight management
│   │   └── config.py                   # Model configuration
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── scheduler.py                # Request scheduler
│   │   ├── sampler.py                  # Sampling strategies
│   │   └── cache_manager.py            # Cache management
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── qwen_tokenizer.py           # Tokenizer wrapper
│   ├── server/
│   │   ├── __init__.py
│   │   ├── api.py                      # FastAPI server
│   │   └── handlers.py                 # Request handlers
│   └── utils/
│       ├── __init__.py
│       └── logging.py                  # Logging utilities
│
├── tests/
│   ├── unit/
│   │   ├── test_kernels.py             # Kernel tests
│   │   ├── test_cache.py               # Cache tests
│   │   └── test_scheduler.py           # Scheduler tests
│   ├── integration/
│   │   ├── test_end_to_end.py         # E2E tests
│   │   └── test_inference.py           # Inference tests
│   └── performance/
│       ├── benchmark_kernels.py        # Kernel benchmarks
│       └── benchmark_e2e.py            # E2E benchmarks
│
├── examples/
│   ├── simple_generation.py            # Basic generation example
│   ├── batch_processing.py             # Batch processing example
│   └── streaming_response.py           # Streaming example
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile                  # Production Docker image
│   │   └── docker-compose.yml          # Multi-service setup
│   └── kubernetes/
│       ├── deployment.yaml             # K8s deployment
│       └── service.yaml                # K8s service
│
├── config/
│   ├── model_config.json               # Model configuration
│   ├── inference_config.json           # Inference settings
│   └── server_config.yaml              # Server settings
│
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Python project config
└── README.md                           # Quick start guide
```

---

## Key Concepts

### 1. Paged KV Cache

**Problem**: Traditional KV caches for attention require contiguous memory blocks proportional to sequence length × batch size, leading to memory waste.

**Solution**: Divide KV cache into fixed-size pages (typically 16 tokens per page) that can be allocated/deallocated independently.

**Benefits**:
- Efficient memory utilization
- Reduced fragmentation
- Support for arbitrary batch sizes
- Easier memory reuse across requests

### 2. RadixAttention (PagedAttention with Prefix Sharing)

**Key Features**:
- **Prefix Sharing**: Multiple requests with shared prompt prefixes reuse the same cache blocks
- **Radix Tree Structure**: Hierarchical organization of cache blocks enabling efficient sharing
- **Reference Counting**: Track how many requests use each cache block
- **Dynamic Deallocation**: Automatically free cache blocks when reference count reaches zero

**Use Cases**:
- Batch processing with common prefixes
- Multi-turn conversations with shared system prompts
- Speculative decoding with shared prefix

### 3. Continuous Batching

**Concept**: Dynamically add/remove requests from batches as they complete, rather than waiting for all requests to finish.

**Advantages**:
- Better GPU utilization
- Lower latency for short-running requests
- Higher throughput overall

### 4. FlashAttention for Prefill

**Optimization**: Compute attention in tiles with shared memory optimization for GPU-efficient processing.

**Key Improvements**:
- Reduced memory bandwidth
- Tiled computation with careful I/O scheduling
- Online softmax for numerical stability
- Support for arbitrary sequence lengths

### 5. Decode-Phase Attention

**Optimization**: Single token generation from KV cache.

**Key Improvements**:
- Single query token vs. full KV cache
- Efficient memory access patterns
- Minimal computation per token
- Support for grouped query attention

---

## Development Workflow

### Phase 1: Foundation (Weeks 1-2)

1. Set up Mojo environment and build system
2. Implement basic type definitions and memory utilities
3. Create foundational kernels (RoPE, RMSNorm)
4. Write unit tests for each component

### Phase 2: Attention Kernels (Weeks 3-4)

1. Implement FlashAttention for prefill phase
2. Implement decode-phase attention
3. Add grouped query attention support
4. Benchmark against baselines

### Phase 3: Python Integration (Weeks 5-6)

1. Build model loading and weight management
2. Implement request scheduler with continuous batching
3. Create sampling strategies
4. Add tokenizer integration

### Phase 4: System Integration (Weeks 7-8)

1. Build KV cache manager with RadixAttention
2. Implement request scheduler
3. Create FastAPI inference server
4. Integration testing

### Phase 5: Optimization & Deployment (Weeks 9-10)

1. Performance profiling and optimization
2. Memory optimization
3. Docker containerization
4. Deployment documentation

---

## Performance Targets

### Inference Metrics

- **Prefill Throughput**: > 100K tokens/sec/GPU
- **Decode Throughput**: > 50 tokens/sec (single request)
- **Batch Decode Throughput**: > 5K tokens/sec (batch of 64)
- **E2E Latency**: < 1 second for 512-token prompt + 128 token generation

### Memory Efficiency

- **KV Cache**: < 2 bytes per token per layer (with int8 quantization potential)
- **Model Weights**: Loaded in mixed precision (fp8/fp16)
- **Peak Memory**: < 90% GPU VRAM for batch_size=64

### Scalability

- **Batch Size**: Support up to 256 concurrent requests
- **Sequence Length**: Support up to 32K tokens
- **Multi-GPU**: Tensor parallelism support (optional extension)

---

## Next Steps

1. **Read** `02_MOJO_KERNEL_GUIDE.md` for step-by-step kernel implementation
2. **Review** `03_PYTHON_INTEGRATION.md` for Python layer setup
3. **Study** `04_ATTENTION_MECHANISMS.md` for attention algorithm details
4. **Follow** `05_KV_CACHE_MANAGEMENT.md` for cache system design
5. **Deploy** using `06_DEPLOYMENT.md`

---

## Resources

### Official Documentation

- [Mojo Programming Language](https://docs.modular.com/mojo/)
- [MAX Framework](https://docs.modular.com/max/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Grouped Query Attention](https://arxiv.org/abs/2305.13245)
- [PagedAttention](https://arxiv.org/abs/2309.06180)

### Recommended Reading

- **Transformers Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT-3 Details**: "Language Models are Unsupervised Multitask Learners"
- **LLM Inference**: "Efficient Memory Management for Large Language Model Serving"

---

## Support & Community

- **Issues**: Report bugs or feature requests
- **Discussions**: Ask questions and share ideas
- **Contributing**: Contributions welcome! See CONTRIBUTING.md

---

## License

This project is released under the MIT License. See LICENSE file for details.
