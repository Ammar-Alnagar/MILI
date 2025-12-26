# MILI: Machine Learning Inference Lattice for Qwen3

## Overview

This guide walks you through building a **MILI (Machine Learning Inference Lattice)** inference server for the **Qwen3 language model**. MILI provides a production-ready inference system that leverages HuggingFace transformers for efficient model serving with GPU acceleration.

This project focuses on **practical deployment** of large language models, providing a complete inference pipeline from model loading to API serving, while maintaining high performance and scalability.

### What You'll Build

By following this guide, you will have:

1. **HuggingFace Integration** for loading and running Qwen3 models
2. **GPU-Accelerated Inference** with automatic device management
3. **FastAPI Server** with RESTful endpoints for text generation
4. **Tokenizer Integration** using official Qwen3 tokenizers
5. **Production-Ready Deployment** with health checks and monitoring

---

## Target Architecture: Qwen3

### Model Characteristics

- **Official Qwen3 Model**: Uses the actual Qwen/Qwen3-0.6B model from HuggingFace
- **Decoder-Only Transformer** architecture with GQA and RoPE
- **Optimized for Inference**: Pre-trained and optimized by Alibaba Cloud
- **Standard Interface**: Compatible with transformers library
- **GPU Acceleration**: Automatic CUDA support when available
- **Vocabulary Size**: 151,936 tokens
- **Context Length**: Up to 40,960 tokens
- **Model Size**: 0.6B parameters (lightweight for demos and development)

### Current Configuration

The server currently loads Qwen/Qwen3-0.6B with these specifications:

```
MODEL CONFIG (Qwen3-0.6B):
- hidden_size: 1024
- intermediate_size: 3072
- num_attention_heads: 16
- num_key_value_heads: 8  # GQA enabled
- num_hidden_layers: 28
- vocab_size: 151936
- max_position_embeddings: 40960
- rope_theta: 1000000.0
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
│  │  - RESTful API endpoints                                 │   │
│  │  - Request validation and response formatting            │   │
│  │  - Health checks and monitoring                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         HuggingFace Transformers Layer                  │   │
│  │  - AutoTokenizer for text encoding/decoding             │   │
│  │  - AutoModelForCausalLM for inference                    │   │
│  │  - Automatic device management (CPU/GPU)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Model Management                            │   │
│  │  - Automatic model downloading and caching              │   │
│  │  - Weight loading and optimization                       │   │
│  │  - Memory-efficient inference                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│              GPU/CPU (NVIDIA CUDA or CPU fallback)               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Dependencies

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
  - Minimum: RTX 3060 (12GB VRAM)
  - Recommended: RTX 4070 or higher
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 16GB+ for model loading and inference
- **Storage**: 5GB+ for model weights and dependencies

### Software Prerequisites

Before starting, ensure you have:

1. **Python 3.8+**
    ```bash
    python --version  # Should be 3.8 or higher
    ```

2. **CUDA Toolkit** (for GPU acceleration, optional)
    ```bash
    nvidia-smi  # Verify CUDA installation (optional)
    ```

3. **Git** (for cloning repositories)
    ```bash
    git --version
    ```

### Python Dependencies

```
torch>=2.0.0          # PyTorch for tensor operations
transformers>=4.35.0  # HuggingFace transformers for model loading
fastapi>=0.104.0      # FastAPI for the web server
uvicorn>=0.24.0       # ASGI server for FastAPI
pydantic>=2.0.0       # Data validation
numpy>=1.24.0         # Numerical operations
pytest>=7.4.0         # Testing framework (optional)
```

---

## Project Organization

```
mili_qwen3/
├── config/
│   ├── model_config.json               # Model configuration (legacy)
│   └── inference_config.json           # Inference settings (legacy)
│
├── docs/
│   ├── 01_PROJECT_OVERVIEW.md          # This file
│   ├── 02_MOJO_KERNEL_GUIDE.md         # Mojo kernel development (legacy)
│   ├── 03_PYTHON_INTEGRATION.md        # Python layer setup
│   ├── 04_ATTENTION_MECHANISMS.md      # Attention docs (legacy)
│   ├── 05_KV_CACHE_MANAGEMENT.md       # Cache docs (legacy)
│   ├── 06_DEPLOYMENT.md                # Deployment guide
│   ├── 07_ADVANCED_OPTIMIZATION.md     # Optimization guide (legacy)
│   ├── 08_TROUBLESHOOTING_AND_DEBUGGING.md
│   ├── 09_API_REFERENCE.md             # API documentation
│   └── 10_BEST_PRACTICES_AND_PATTERNS.md
│
├── examples/
│   └── basic_inference.py              # Basic generation example
│
├── mojo_kernels/                       # Legacy Mojo kernels (not used)
│   ├── core/
│   │   ├── activations.🔥
│   │   ├── attention.🔥
│   │   ├── normalization.🔥
│   │   └── rope.🔥
│   ├── memory/
│   │   └── kv_cache.🔥
│   ├── utils/
│   │   └── types.🔥
│   ├── build.sh
│   └── test_simple.mojo
│
├── python_layer/                       # Legacy Python components
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inference_engine.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── kv_cache_manager.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── qwen_model.py
│   │   └── weight_loader.py
│   ├── server/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── qwen_tokenizer.py
│   └── utils/
│       └── __init__.py
│
├── tests/
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_inference.py
│   ├── performance/
│   │   └── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_tokenizer.py
│   └── __init__.py
│
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Python project config
├── server.py                           # Main inference server
├── test_real_weights.py               # Weight loading test
├── verify_implementation.py           # Implementation verification
├── verify_simple.py                   # Simple verification
├── IMPLEMENTATION_GUIDE.md           # Implementation guide
├── INDEX.md                           # Project index
├── STRUCTURE.md                       # Project structure
├── DELIVERABLES.txt                   # Deliverables checklist
└── README.md                          # Quick start guide
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

### Phase 1: Setup and Basic Server (Day 1)

1. Set up Python environment and install dependencies
2. Clone the MILI repository and explore the codebase
3. Run the basic inference server locally
4. Test the API endpoints with sample requests

### Phase 2: Understanding the Codebase (Day 2)

1. Examine the server.py implementation
2. Understand how transformers integration works
3. Review the configuration and model loading
4. Test different generation parameters

### Phase 3: Customization and Extension (Day 3-4)

1. Modify server configuration for different models
2. Add custom preprocessing/postprocessing
3. Implement additional API endpoints
4. Add monitoring and logging features

### Phase 4: Production Deployment (Day 5)

1. Containerize the application with Docker
2. Set up production server configuration
3. Implement health checks and monitoring
4. Deploy to cloud infrastructure

---

## Performance Targets

### Inference Metrics (Qwen3-0.6B)

- **Prefill Throughput**: 500-2000 tokens/sec (depends on hardware)
- **Decode Throughput**: 20-100 tokens/sec (single request)
- **E2E Latency**: 2-10 seconds for 512-token prompt + 128 token generation
- **Memory Usage**: ~2-4GB GPU VRAM, ~4-8GB system RAM

### Scalability

- **Concurrent Requests**: 1-10 simultaneous requests (depending on hardware)
- **Sequence Length**: Up to 4096 tokens (model limit)
- **Model Size**: Easily extensible to larger Qwen models

### Scalability

- **Batch Size**: Support up to 256 concurrent requests
- **Sequence Length**: Support up to 32K tokens
- **Multi-GPU**: Tensor parallelism support (optional extension)

---

## Next Steps

1. **Read** `03_PYTHON_INTEGRATION.md` for server setup and usage
2. **Review** `06_DEPLOYMENT.md` for production deployment
3. **Check** `09_API_REFERENCE.md` for complete API documentation
4. **Follow** `08_TROUBLESHOOTING_AND_DEBUGGING.md` for common issues
5. **Explore** the examples in the `examples/` directory

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
