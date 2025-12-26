# MILI: Machine Learning Inference Lattice for Qwen-Family

A comprehensive, hands-on guide to building a high-performance LLM inference engine in Mojo and Python.

##  Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA 12.0+ (A100, H100, or RTX 4090 recommended)
- **Mojo SDK**: Latest version from Modular
- **Python**: 3.10 or higher
- **CUDA Toolkit**: 12.0+

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd mili_qwen3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build Mojo kernels
cd mojo_kernels
bash build.sh
cd ..
```

### Run Inference Server

```bash
# Start server
python server.py

# In another terminal, test it:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

---

##  Documentation Structure

This project is organized as a progressive learning guide:

### 1. **[Project Overview](mili_qwen3/docs/01_PROJECT_OVERVIEW.md)** - START HERE
    - High-level architecture
    - System design principles
    - Project organization
    - Prerequisites and setup

### 2. **[Mojo Kernel Guide](mili_qwen3/docs/02_MOJO_KERNEL_GUIDE.md)** - GPU Kernels (Legacy)
    - Foundation & setup
    - RoPE implementation
    - RMSNorm kernels
    - SwiGLU activation
    - FlashAttention prefill
    - Decode-phase attention
    - Memory management

### 3. **[Python Integration Guide](mili_qwen3/docs/03_PYTHON_INTEGRATION.md)** - Python Layer
    - Model architecture & config
    - Weight loading
    - Tokenization (tiktoken)
    - Request scheduler (continuous batching)
    - Sampling strategies
    - Model class integration

### 4. **[Attention Mechanisms](mili_qwen3/docs/04_ATTENTION_MECHANISMS.md)** - Deep Dive
    - Scaled dot-product attention
    - Grouped Query Attention (GQA)
    - FlashAttention optimization
    - Decode-phase optimization
    - Multi-request attention
    - Performance benchmarks

### 5. **[KV Cache Management](mili_qwen3/docs/05_KV_CACHE_MANAGEMENT.md)** - Memory Efficiency
    - Paged KV cache
    - RadixAttention for prefix sharing
    - Reference counting
    - Allocation strategies
    - Eviction policies
    - Integration with inference loop

### 6. **[Deployment Guide](mili_qwen3/docs/06_DEPLOYMENT.md)** - Production Ready
    - FastAPI server setup
    - Docker containerization
    - Kubernetes deployment
    - GPU optimization
    - Monitoring & metrics
    - Load testing

### 7. **[Advanced Optimization](mili_qwen3/docs/07_ADVANCED_OPTIMIZATION.md)** - Performance Tuning
    - Kernel optimization techniques
    - Memory bandwidth optimization
    - Parallel processing strategies

### 8. **[Troubleshooting and Debugging](mili_qwen3/docs/08_TROUBLESHOOTING_AND_DEBUGGING.md)** - Common Issues
    - Environment setup problems
    - Kernel compilation issues
    - Python integration bugs
    - Server deployment issues
    - Performance bottlenecks

### 9. **[API Reference](mili_qwen3/docs/09_API_REFERENCE.md)** - Complete API Docs
    - Server endpoints
    - Request/response formats
    - Configuration options
    - Error handling

### 10. **[Best Practices and Patterns](mili_qwen3/docs/10_BEST_PRACTICES_AND_PATTERNS.md)** - Production Patterns
    - Code organization
    - Testing strategies
    - Deployment best practices
    - Performance monitoring

---

## Project Structure

```
mili_qwen3/
├── config/                         # Configuration files
│   ├── inference_config.json       # Inference settings
│   └── model_config.json           # Model configuration
│
├── docs/                           # Documentation
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_MOJO_KERNEL_GUIDE.md
│   ├── 03_PYTHON_INTEGRATION.md
│   ├── 04_ATTENTION_MECHANISMS.md
│   ├── 05_KV_CACHE_MANAGEMENT.md
│   ├── 06_DEPLOYMENT.md
│   ├── 07_ADVANCED_OPTIMIZATION.md
│   ├── 08_TROUBLESHOOTING_AND_DEBUGGING.md
│   ├── 09_API_REFERENCE.md
│   └── 10_BEST_PRACTICES_AND_PATTERNS.md
│
├── examples/                       # Example scripts
│   └── basic_inference.py          # Basic inference example
│
├── mojo_kernels/                   # Mojo kernel implementations (legacy)
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
├── python_layer/                   # Python components
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
├── tests/                          # Test suites
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
├── DELIVERABLES.txt                # Project deliverables
├── IMPLEMENTATION_GUIDE.md         # Implementation guide
├── INDEX.md                        # Project index
├── pyproject.toml                  # Python project config
├── requirements.txt                # Python dependencies
├── server.py                       # Main inference server
├── STRUCTURE.md                    # Project structure
├── test_project.py                 # Project test script
├── test_qwen3_local.py             # Local Qwen3 test
├── test_real_weights.py            # Weight loading test
├── verify_implementation.py        # Implementation verification
└── verify_simple.py                # Simple verification
```

---

##  Learning Path

### Week 1-2: Foundation
- [ ] Read Project Overview
- [ ] Set up development environment
- [ ] Review Mojo SDK basics
- [ ] Understand transformer architecture

### Week 3-4: Kernels
- [ ] Study Mojo Kernel Guide
- [ ] Implement RoPE kernel
- [ ] Implement RMSNorm kernel
- [ ] Implement SwiGLU activation
- [ ] Write unit tests

### Week 5-6: Attention
- [ ] Study Attention Mechanisms guide
- [ ] Understand FlashAttention algorithm
- [ ] Implement prefill attention
- [ ] Implement decode attention
- [ ] Benchmark implementations

### Week 7-8: Python Layer
- [ ] Study Python Integration guide
- [ ] Implement model config
- [ ] Implement weight loader
- [ ] Implement tokenizer wrapper
- [ ] Implement request scheduler

### Week 9-10: Memory & Cache
- [ ] Study KV Cache Management guide
- [ ] Implement paged KV cache
- [ ] Implement RadixAttention
- [ ] Implement allocation strategies
- [ ] Integration testing

### Week 11-12: Deployment
- [ ] Study Deployment guide
- [ ] Build FastAPI server
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Load testing & optimization

---

##  Development Commands

### Building

```bash
# Build all Mojo kernels
cd mojo_kernels && bash build.sh && cd ..

# Build specific kernel with optimization
mojo build -O3 -o lib/core/attention.so core/attention.🔥
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_kernels.py -v

# Run with coverage
pytest tests/ --cov-report=html
```

### Running

```bash
# Development server with auto-reload
python server.py

# Production server (single process for now)
python server.py

# Simple inference script
python examples/simple_generation.py
```

### Benchmarking

```bash
# Benchmark kernels
python tests/performance/benchmark_kernels.py

# End-to-end benchmark
python tests/performance/benchmark_e2e.py

# Load testing
python tests/performance/load_test.py --num-requests 1000 --concurrent 50
```

---

##  Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Prefill Throughput | > 100K tokens/sec |  |
| Decode Throughput | > 50 tokens/sec (1 req) |  |
| Batch Decode | > 5K tokens/sec (batch 64) |  |
| E2E Latency | < 1s (512 + 128 tokens) |  |
| Memory Efficiency | < 90% VRAM (batch 64) |  |

---

##  Docker Quick Start

```bash
# Build image
docker build -t mili:latest -f deployment/docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 mili:latest

# Run with docker-compose
docker-compose -f deployment/docker/docker-compose.yml up
```

---

##  Kubernetes Deployment

```bash
# Deploy
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods
kubectl logs -f deployment/mili-inference

# Port forward
kubectl port-forward svc/mili-inference 8000:80
```

---

##  Example Usage

### Simple Generation

```python
# Using the API server
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "What is the meaning of life?",
    "max_tokens": 100,
    "temperature": 0.7
})

result = response.json()
print(result["generated_text"])

# Or direct Python usage
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("What is the meaning of life?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
    temperature=0.7
)
print(response)
```

### API Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "stream": false
  }'
```

---

##  Key Concepts

### Continuous Batching
Dynamically add/remove requests as they complete, maximizing GPU utilization.

### Paged KV Cache
Divide KV cache into fixed-size pages (16 tokens) for efficient memory management.

### RadixAttention
Share KV cache across multiple requests with common prefixes using a radix tree.

### FlashAttention
Optimize attention with tiled computation and online softmax for 3-4x speedup.

### Grouped Query Attention
Reduce KV cache size by using fewer KV heads than query heads.

---

##  Contributing

Contributions welcome! Areas to help:

- [ ] Implement additional kernels (quantization, fused ops)
- [ ] Optimize existing kernels
- [ ] Add more sampling strategies
- [ ] Improve documentation
- [ ] Add more tests
- [ ] Performance optimizations

---

##  References

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Efficient attention
- [Grouped Query Attention](https://arxiv.org/abs/2305.13245) - KV cache reduction
- [PagedAttention](https://arxiv.org/abs/2309.06180) - Paged KV cache

### Resources
- [Mojo Documentation](https://docs.modular.com/mojo/)
- [MAX Framework](https://docs.modular.com/max/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [vLLM](https://github.com/lm-sys/vllm) - Reference implementation

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Support

- **Issues**: Report bugs or feature requests
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check docs/ for detailed guides

---

##  Acknowledgments

This project builds upon research from:
- Modular AI team (Mojo language)
- DeepSpeed team (optimization techniques)
- vLLM team (inference system design)
- HuggingFace community (models and tools)

---

**Happy inferencing! **

For a detailed walkthrough, start with [01_PROJECT_OVERVIEW.md](mili_qwen3/docs/01_PROJECT_OVERVIEW.md).
