# MILI: Machine Learning Inference Lattice for Qwen3

A comprehensive, hands-on guide to building a high-performance LLM inference system in Mojo and Python.

## 🚀 Quick Start

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
python -m uvicorn python_layer.server.api:app --reload --port 8000

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

## 📚 Documentation Structure

This project is organized as a progressive learning guide:

### 1. **[Project Overview](docs/01_PROJECT_OVERVIEW.md)** ⭐ START HERE
   - High-level architecture
   - System design principles
   - Project organization
   - Prerequisites and setup

### 2. **[Mojo Kernel Guide](docs/02_MOJO_KERNEL_GUIDE.md)** - GPU Kernels
   - Foundation & setup
   - RoPE implementation
   - RMSNorm kernels
   - SwiGLU activation
   - FlashAttention prefill
   - Decode-phase attention
   - Memory management

### 3. **[Python Integration Guide](docs/03_PYTHON_INTEGRATION.md)** - Python Layer
   - Model architecture & config
   - Weight loading
   - Tokenization (tiktoken)
   - Request scheduler (continuous batching)
   - Sampling strategies
   - Model class integration

### 4. **[Attention Mechanisms](docs/04_ATTENTION_MECHANISMS.md)** - Deep Dive
   - Scaled dot-product attention
   - Grouped Query Attention (GQA)
   - FlashAttention optimization
   - Decode-phase optimization
   - Multi-request attention
   - Performance benchmarks

### 5. **[KV Cache Management](docs/05_KV_CACHE_MANAGEMENT.md)** - Memory Efficiency
   - Paged KV cache
   - RadixAttention for prefix sharing
   - Reference counting
   - Allocation strategies
   - Eviction policies
   - Integration with inference loop

### 6. **[Deployment Guide](docs/06_DEPLOYMENT.md)** - Production Ready
   - FastAPI server setup
   - Docker containerization
   - Kubernetes deployment
   - GPU optimization
   - Monitoring & metrics
   - Load testing

---

## 📁 Project Structure

```
mili_qwen3/
├── docs/                           # Complete guides
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_MOJO_KERNEL_GUIDE.md
│   ├── 03_PYTHON_INTEGRATION.md
│   ├── 04_ATTENTION_MECHANISMS.md
│   ├── 05_KV_CACHE_MANAGEMENT.md
│   └── 06_DEPLOYMENT.md
│
├── mojo_kernels/                   # GPU kernel implementations
│   ├── core/                       # Core compute kernels
│   │   ├── attention.🔥           # FlashAttention, etc.
│   │   ├── rope.🔥                # Rotary Position Embeddings
│   │   ├── activations.🔥         # SwiGLU, GELU, etc.
│   │   └── normalization.🔥       # RMSNorm
│   ├── memory/                     # Memory management
│   │   ├── kv_cache.🔥            # Paged KV cache
│   │   └── allocator.🔥           # Memory allocator
│   ├── utils/                      # Utilities
│   │   ├── types.🔥               # Type definitions
│   │   └── helpers.🔥             # Helper functions
│   └── build.sh                    # Build script
│
├── python_layer/                   # Python inference layer
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py               # Qwen3Config
│   │   ├── weight_loader.py        # Weight management
│   │   └── qwen3_model.py          # Model class
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── scheduler.py            # Request scheduler
│   │   ├── sampler.py              # Sampling strategies
│   │   └── cache_manager.py        # Cache manager
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── qwen_tokenizer.py       # Tokenizer wrapper
│   ├── server/
│   │   ├── __init__.py
│   │   ├── api.py                  # FastAPI server
│   │   └── handlers.py             # Request handlers
│   └── utils/
│       ├── __init__.py
│       └── logging.py              # Logging utilities
│
├── tests/
│   ├── unit/                       # Unit tests
│   │   ├── test_kernels.py
│   │   ├── test_cache.py
│   │   └── test_scheduler.py
│   ├── integration/                # Integration tests
│   │   ├── test_end_to_end.py
│   │   └── test_inference.py
│   └── performance/                # Performance benchmarks
│       ├── benchmark_kernels.py
│       └── benchmark_e2e.py
│
├── examples/
│   ├── simple_generation.py        # Basic generation
│   ├── batch_processing.py         # Batch processing
│   └── streaming_response.py       # Streaming generation
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
│       ├── deployment.yaml
│       └── service.yaml
│
├── config/
│   ├── model_config.json           # Model configuration
│   ├── inference_config.json       # Inference settings
│   └── server_config.yaml          # Server configuration
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Python project config
├── LICENSE
└── README.md                       # This file
```

---

## 🎯 Learning Path

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

## 🔧 Development Commands

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
pytest tests/ --cov=python_layer --cov-report=html
```

### Running

```bash
# Development server with auto-reload
python -m uvicorn python_layer.server.api:app --reload --port 8000

# Production server
python -m uvicorn python_layer.server.api:app --workers 4 --port 8000

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

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Prefill Throughput | > 100K tokens/sec | 🎯 |
| Decode Throughput | > 50 tokens/sec (1 req) | 🎯 |
| Batch Decode | > 5K tokens/sec (batch 64) | 🎯 |
| E2E Latency | < 1s (512 + 128 tokens) | 🎯 |
| Memory Efficiency | < 90% VRAM (batch 64) | 🎯 |

---

## 🐳 Docker Quick Start

```bash
# Build image
docker build -t mili:latest -f deployment/docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 mili:latest

# Run with docker-compose
docker-compose -f deployment/docker/docker-compose.yml up
```

---

## ☸️ Kubernetes Deployment

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

## 🧪 Example Usage

### Simple Generation

```python
from python_layer.model.qwen3_model import Qwen3Model
from python_layer.model.config import Qwen3Config
from python_layer.model.weight_loader import WeightLoader

# Setup
config = Qwen3Config.from_pretrained("Qwen/Qwen3-7B")
weight_loader = WeightLoader("./models/Qwen3-7B", config)
weight_loader.load_from_huggingface("Qwen/Qwen3-7B")
model = Qwen3Model(config, weight_loader)

# Generate
response = model.generate(
    prompt="What is the meaning of life?",
    max_tokens=100,
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

## 📝 Key Concepts

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

## 🤝 Contributing

Contributions welcome! Areas to help:

- [ ] Implement additional kernels (quantization, fused ops)
- [ ] Optimize existing kernels
- [ ] Add more sampling strategies
- [ ] Improve documentation
- [ ] Add more tests
- [ ] Performance optimizations

---

## 📚 References

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

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support

- **Issues**: Report bugs or feature requests
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check docs/ for detailed guides

---

## 🎓 Acknowledgments

This project builds upon research from:
- Modular AI team (Mojo language)
- DeepSpeed team (optimization techniques)
- vLLM team (inference system design)
- HuggingFace community (models and tools)

---

**Happy inferencing! 🚀**

For a detailed walkthrough, start with [01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md).
