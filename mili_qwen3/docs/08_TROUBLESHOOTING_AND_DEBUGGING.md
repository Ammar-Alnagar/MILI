# Troubleshooting and Debugging Guide for MILI

## Overview

This guide provides solutions to common issues encountered during MILI development, deployment, and operation.

---

## Part 1: Environment and Setup Issues

### Issue 1.1: Mojo SDK Installation Fails

**Symptoms**: 
```
error: Failed to install Mojo SDK
error: CUDA not found
```

**Solutions**:

```bash
# 1. Verify CUDA installation
nvidia-smi

# 2. Check CUDA toolkit version
nvcc --version

# 3. Install/update CUDA (if needed)
# For Ubuntu:
sudo apt-get install nvidia-cuda-toolkit

# 4. Reinstall Mojo SDK
curl https://docs.modular.com/mojo/install.sh | bash

# 5. Add to PATH
export PATH="/home/user/.modular/pkg/mojo:$PATH"

# 6. Verify installation
mojo --version
```

### Issue 1.2: Python Virtual Environment Issues

**Symptoms**:
```
Error: Python version not supported
Error: pip install fails with dependency conflicts
```

**Solutions**:

```bash
# 1. Ensure Python 3.10+
python3 --version

# 2. Create fresh venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install with constraints
pip install -r requirements.txt --no-cache-dir

# 5. Check for conflicts
pip check

# 6. If still issues, use specific versions
pip install torch==2.0.0 --no-cache-dir
pip install transformers==4.35.0 --no-cache-dir
```

### Issue 1.3: CUDA Out of Memory During Setup

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

```python
# 1. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 2. Check GPU memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 3. Reduce batch size
batch_size = 32  # Reduce from 64

# 4. Use smaller model
# The Qwen3-0.6B is already the smallest available
# For even smaller models, consider other architectures

# 5. Enable memory efficient mode
import torch.cuda
torch.cuda.set_per_process_memory_fraction(0.8)

# 6. Use gradient checkpointing (if applicable)
# model.gradient_checkpointing_enable()
```

---

## Part 2: Kernel Compilation Issues

### Issue 2.1: Mojo Kernel Compilation Fails

**Symptoms**:
```
error: Mojo compilation failed
error: MLIR lowering failed
error: Type mismatch in kernel
```

**Solutions**:

```bash
# 1. Enable verbose output
mojo build -v core/rope. -o lib/rope.so

# 2. Check for syntax errors
# Review the specific line mentioned in error

# 3. Verify type declarations
# Ensure all types are properly annotated

# 4. Simplify kernel step-by-step
# Start with minimal version and add features

# 5. Use debug symbols
mojo build -g core/rope. -o lib/rope.so

# 6. Check MAX version compatibility
modular info
```

### Issue 2.2: CUDA Kernel Launch Fails

**Symptoms**:
```
CUDA_ERROR_INVALID_GRID_DIMENSION
CUDA_ERROR_LAUNCH_TIMEOUT
```

**Solutions**:

```mojo
# 1. Check grid/block dimensions
fn validate_kernel_config(
    grid_size: Int,
    block_size: Int
) -> Bool:
    # Max grid: 65535 in each dimension (sm_70+)
    # Max block: 1024 threads total
    
    let max_threads = 1024
    let max_grid = 65535
    
    if block_size > max_threads:
        print("ERROR: block_size > 1024")
        return False
    
    if grid_size > max_grid:
        print("ERROR: grid_size > 65535")
        return False
    
    return True

# 2. Typical safe configuration
let block_size = 256  # Multiple of 32 (warp size)
let grid_size = (total_elements + block_size - 1) // block_size

# 3. Use smaller blocks for debugging
let debug_block_size = 32
```

### Issue 2.3: Kernel Produces Wrong Results

**Symptoms**:
```
AssertionError: Output doesn't match expected values
NaN or Inf in output
```

**Solutions**:

```python
# 1. Test with CPU reference
import torch

def cpu_rope(q, freqs):
    """CPU reference implementation."""
    real = torch.cos(freqs)
    imag = torch.sin(freqs)
    
    # Rotate in 2D
    q_rot = torch.zeros_like(q)
    q_rot[..., 0::2] = q[..., 0::2] * real - q[..., 1::2] * imag
    q_rot[..., 1::2] = q[..., 0::2] * imag + q[..., 1::2] * real
    
    return q_rot

# 2. Compare outputs
gpu_output = gpu_rope_kernel(q, freqs)
cpu_output = cpu_rope(q, freqs)

diff = torch.abs(gpu_output - cpu_output).max()
print(f"Max difference: {diff}")

if diff > 1e-4:
    print("Kernel produces incorrect results")
    # Debug step-by-step

# 3. Check for numerical issues
print(f"Output range: [{gpu_output.min()}, {gpu_output.max()}]")
print(f"NaN count: {torch.isnan(gpu_output).sum()}")
print(f"Inf count: {torch.isinf(gpu_output).sum()}")

# 4. Add intermediate assertions
# Verify RMS normalization preserves norm
input_norm = torch.norm(input_tensor)
output_norm = torch.norm(output_tensor)
assert torch.allclose(input_norm, output_norm, rtol=1e-3)
```

---

## Part 3: Python Integration Issues

### Issue 3.1: Model Loading Fails

**Symptoms**:
```
FileNotFoundError: Model files not found
KeyError: Weight key not found
```

**Solutions**:

```python
# 1. Check model directory
import os
from pathlib import Path

model_dir = Path("./models/Qwen3-7B")
if not model_dir.exists():
    print(f"ERROR: Model directory {model_dir} not found")
    print("Download model: https://huggingface.co/Qwen/Qwen3-7B")

# 2. List available files
if model_dir.exists():
    files = list(model_dir.glob("**/*"))
    print("Available files:")
    for f in files:
        print(f"  {f.name}")

# 3. Download from HuggingFace
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-7B",
    cache_dir="./models",
    resume_download=True
)

# 4. Handle missing keys gracefully
try:
    weight_loader.load_from_safetensors()
except KeyError as e:
    print(f"Missing weight key: {e}")
    print("Try using from_pretrained instead")
    weight_loader.load_from_huggingface("Qwen/Qwen3-7B")

# 5. Verify weight loading
print(f"Loaded {len(weight_loader.weights)} weight tensors")
for name, tensor in list(weight_loader.weights.items())[:5]:
    print(f"  {name}: {tensor.shape}")
```

### Issue 3.2: Request Scheduler Hangs

**Symptoms**:
```
Process hangs after requests submitted
CPU usage stays at 0%
```

**Solutions**:

```python
# 1. Check if scheduler is running
scheduler = ContinuousBatchScheduler()

# Add timeout to request processing
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Scheduler processing timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    batch, batch_type = scheduler.get_next_batch()
    print(f"Got batch: {len(batch)} requests, type={batch_type}")
except TimeoutError:
    print("ERROR: Scheduler timeout - may be hung")

signal.alarm(0)  # Disable alarm

# 2. Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

scheduler = ContinuousBatchScheduler()
req_id = scheduler.add_request([1, 2, 3, 4, 5])
print(f"Added request: {req_id}")

status = scheduler.get_request_status(req_id)
print(f"Request status: {status}")

# 3. Check for deadlocks
# Verify no circular dependencies between requests
# Check KV cache allocation doesn't deadlock

# 4. Use process timeout
import multiprocessing
import time

def run_scheduler_with_timeout():
    scheduler = ContinuousBatchScheduler()
    # ... process requests
    
if __name__ == "__main__":
    p = multiprocessing.Process(target=run_scheduler_with_timeout)
    p.start()
    p.join(timeout=60)
    
    if p.is_alive():
        print("ERROR: Scheduler timeout")
        p.terminate()
```

### Issue 3.3: Tokenizer Produces Wrong Tokens

**Symptoms**:
```
Decoded text doesn't match input
Token IDs out of vocabulary range
```

**Solutions**:

```python
# 1. Verify tokenizer initialization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(f"Vocab size: {tokenizer.vocab_size}")

# 2. Test encode/decode round-trip
test_text = "Hello, world!"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Original:  {test_text}")
print(f"Tokens:    {tokens}")
print(f"Decoded:   {decoded}")

# Check for exact match (may differ due to whitespace handling)
if test_text.lower() in decoded.lower():
    print(" Round-trip successful")
else:
    print(" Round-trip failed")

# 3. Check token ID ranges
vocab_size = tokenizer.vocab_size
for token_id in tokens:
    if token_id < 0 or token_id >= vocab_size:
        print(f"ERROR: Token ID {token_id} out of range [0, {vocab_size})")

# 4. Compare with reference tokenizer
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ref_tokens = enc.encode(test_text)

if tokens != ref_tokens:
    print("WARNING: Different encoding than reference")
    print(f"  Our tokens: {tokens}")
    print(f"  Ref tokens: {ref_tokens}")

# 5. Handle special tokens
special_tokens = {
    "<|im_start|>": -1,  # Not in vocab
    "<|im_end|>": -2,
}

for token_str, token_id in special_tokens.items():
    try:
        encoded = tokenizer.encode(token_str)
        print(f"{token_str}: {encoded}")
    except:
        print(f"WARNING: Cannot encode {token_str}")
```

---

## Part 4: FastAPI Server Issues

### Issue 4.1: Server Fails to Start

**Symptoms**:
```
error: Address already in use
error: CORS error
error: 404 Not Found
```

**Solutions**:

```bash
# 1. Address already in use
# Find and kill process using port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port
python server.py

# 2. CORS issues
# Add CORS middleware in api.py:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Check app module
python -c "from server import app; print('App loaded successfully')"

# 4. Test server startup
python server.py

# 5. Check dependencies
pip list | grep fastapi
pip list | grep uvicorn
```

### Issue 4.2: High Latency on Requests

**Symptoms**:
```
Response time > 1 second
GPU not fully utilized
```

**Solutions**:

```python
# 1. Enable profiling
import time
# Current implementation handles single requests only
# Batching support can be added in future versions

import time

# Measure single request processing time
start = time.time()
# Add your request processing code here
processing_time = time.time() - start

print(f"Request processing time: {processing_time*1000:.2f}ms")
print(f"Batch size: {len(batch)}")
print(f"Throughput: {len(batch)/batch_time:.0f} req/sec")

# 2. Increase batch size
scheduler = ContinuousBatchScheduler(
    max_batch_size=128,  # Increase from 64
    prefill_batch_size=64,
    decode_batch_size=256
)

# 3. Enable PyTorch memory optimization
import torch
torch.cuda.empty_cache()  # Clear GPU cache
torch.cuda.set_per_process_memory_fraction(0.8)  # Limit memory usage

# 4. Use smaller model (Qwen3-0.6B is already optimized)
# For memory-constrained systems, consider CPU-only operation

# 5. Profile FastAPI endpoint
import cProfile
import pstats

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Call endpoint
    # ...
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# 6. Check GPU utilization
import torch
def monitor_gpu():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    print(f"GPU Utilization: {torch.cuda.utilization():.1f}%")  # Requires nvidia-ml-py
```

### Issue 4.3: Out of Memory During Inference

**Symptoms**:
```
RuntimeError: CUDA out of memory during inference
```

**Solutions**:

```python
# 1. Reduce batch size
max_batch_size = 32  # Was 64

# 2. Enable PyTorch memory optimization
import torch
torch.cuda.set_per_process_memory_fraction(0.7)  # Limit memory usage

# 3. Use model in half precision if supported
# model = model.half()  # Convert to float16 (experimental)

# 4. Enable gradient checkpointing (if training)
# model.gradient_checkpointing_enable()

# 5. Clear cache periodically
import torch

def clear_cache_periodically():
    torch.cuda.empty_cache()

# 6. Monitor memory in real-time
import threading
import time

def memory_monitor():
    while True:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        time.sleep(5)

monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
monitor_thread.start()

# 7. Fallback to CPU for large requests
def inference_with_fallback(model, input_ids, device="cuda"):
    try:
        return model(input_ids.to(device))
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("WARNING: GPU OOM, falling back to CPU")
            torch.cuda.empty_cache()
            return model(input_ids.cpu()).to(device)
        raise
```

---

## Part 5: Performance and Optimization Issues

### Issue 5.1: Low GPU Utilization

**Symptoms**:
```
GPU utilization < 50%
GPU-Util in nvidia-smi shows low %
```

**Solutions**:

```python
# 1. Check bottleneck with profiling
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run inference
    model(input_ids)

# Print bottleneck operations
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 2. Increase batch size
batch_size = 128  # Increase concurrency

# 3. Implement continuous batching
scheduler = ContinuousBatchScheduler(
    max_batch_size=256,
    decode_batch_size=512
)

# 4. Use tensor parallelism
# Distribute computation across multiple GPUs

# 5. Enable compute intensity optimization
# Use fused kernels where possible
# Reduce memory bandwidth pressure

# 6. Pin CPU threads to NUMA nodes (multi-socket)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
```

### Issue 5.2: Slow Prefill Performance

**Symptoms**:
```
Prefill throughput < 50K tokens/sec
Initial prompt processing slow
```

**Solutions**:

```python
# 1. Use FlashAttention
# Verify it's enabled in model config
config.use_flash_attention_2 = True

# 2. Increase prefill batch size
scheduler = ContinuousBatchScheduler(
    prefill_batch_size=128  # Increase from 32
)

# 3. Profile attention kernel
# Check if using optimized implementation

# 4. Use kernel fusion
# Fuse RoPE + Attention for better cache locality

# 5. Check memory bandwidth
# Ensure coalesced memory access

# 6. Benchmark different kernel implementations
def benchmark_attention():
    q = torch.randn(1, 32, 512, 64).cuda()
    k = torch.randn(1, 32, 512, 64).cuda()
    v = torch.randn(1, 32, 512, 64).cuda()
    
    import time
    
    # Standard attention
    start = time.time()
    for _ in range(100):
        scores = torch.matmul(q, k.transpose(-2, -1))
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"Standard attention: {std_time/100*1000:.3f}ms")
    # Compare with FlashAttention
```

### Issue 5.3: Slow Decode Performance

**Symptoms**:
```
Decode throughput < 20 tokens/sec
Token generation slow
```

**Solutions**:

```python
# 1. Use decode-specific optimization
# Smaller kernel for single-token generation

# 2. Optimize KV cache access
# Ensure paging is working efficiently

# 3. Use memory pooling
pool = GPUMemoryPool(total_size_gb=20)

# 4. Increase decode batch size
scheduler = ContinuousBatchScheduler(
    decode_batch_size=512  # Increase concurrency
)

# 5. Profile KV cache access patterns
# Verify optimal memory layout

# 6. Test different attention implementations
def benchmark_decode_attention():
    q = torch.randn(1, 32, 1, 64).cuda()  # Single token
    k = torch.randn(1, 32, 4096, 64).cuda()  # Full cache
    v = torch.randn(1, 32, 4096, 64).cuda()
    
    import time
    start = time.time()
    for _ in range(1000):
        scores = torch.matmul(q, k.transpose(-2, -1))
    torch.cuda.synchronize()
    time_taken = time.time() - start
    
    print(f"Decode attention: {time_taken/1000*1000:.3f}ms per token")
```

---

## Part 6: Docker and Kubernetes Issues

### Issue 6.1: Docker Build Fails

**Symptoms**:
```
error: Dockerfile.RUN: command failed
error: CUDA driver not found in container
```

**Solutions**:

```bash
# 1. Check Docker is running
docker ps

# 2. Verify base image compatibility
# Ensure CUDA version matches host
docker run --rm nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi

# 3. Build with verbose output
docker build -f deployment/docker/Dockerfile . -t mili:latest --no-cache --progress=plain

# 4. Fix common issues
# - Ensure requirements.txt is in build context
# - Check Mojo installation script is available
# - Verify CUDA toolkit installation

# 5. Build without GPU (if GPU not available)
docker build -f deployment/docker/Dockerfile.cpu . -t mili:cpu

# 6. Test image locally
docker run --gpus all -p 8000:8000 mili:latest
```

### Issue 6.2: Kubernetes Deployment Fails

**Symptoms**:
```
Pod status: CrashLoopBackOff
Pod status: ImagePullBackOff
```

**Solutions**:

```bash
# 1. Check pod status
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# 2. Check image is available
kubectl describe pod <pod-name> | grep Image

# 3. Fix image pull errors
# Ensure image exists in registry
docker tag mili:latest myregistry/mili:latest
docker push myregistry/mili:latest

# Update deployment.yaml with correct image

# 4. Check resource requests
kubectl describe node
# Ensure GPU nodes available

# 5. Fix CrashLoopBackOff
# Check application logs
kubectl logs -f <pod-name>

# 6. Debug pod
kubectl exec -it <pod-name> -- /bin/bash
# Inside pod, check:
# - nvidia-smi works
# - Python environment OK
# - Model files accessible

# 7. Check PVC (if using volumes)
kubectl get pvc
kubectl describe pvc <pvc-name>
```

---

## Part 7: Debugging Checklist

```markdown
## Quick Debugging Checklist

### Environment
- [ ] CUDA is installed and accessible
- [ ] Mojo SDK is installed
- [ ] Python 3.10+ is available
- [ ] Virtual environment is activated
- [ ] All requirements installed: `pip check`

### Kernels
- [ ] Kernels compile without errors: `bash mojo_kernels/build.sh`
- [ ] Kernel outputs match CPU reference
- [ ] No NaN or Inf in outputs
- [ ] Numerical precision acceptable

### Model Loading
- [ ] Model files exist in correct directory
- [ ] Weight keys match expected names
- [ ] Model loads without errors
- [ ] Weight shapes are correct

### Inference
- [ ] Tokenization works correctly
- [ ] Scheduler processes requests
- [ ] Model generates valid tokens
- [ ] Output text is coherent

### Performance
- [ ] GPU utilization > 80%
- [ ] Latency within targets
- [ ] Memory usage reasonable
- [ ] No memory leaks

### Server
- [ ] FastAPI server starts
- [ ] Health check endpoint works
- [ ] Generate endpoint responds
- [ ] Responses are valid JSON

### Deployment
- [ ] Docker image builds
- [ ] Docker image runs locally
- [ ] Kubernetes pod starts
- [ ] Service is accessible
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Model too large | Reduce batch size, quantize model |
| Invalid grid dimension | Wrong kernel config | Fix grid/block dimensions |
| Type mismatch in kernel | Type error in Mojo | Review type annotations |
| Module not found | Missing import | Check Python path |
| Address already in use | Port conflict | Use different port |
| Connection refused | Server not running | Start server first |
| 404 Not Found | Wrong endpoint | Check API routes |
| Timeout | Processing too slow | Optimize or increase timeout |

---

## Getting Help

1. **Check logs**: Always look at detailed error messages
2. **Reproduce**: Create minimal reproducible example
3. **Search**: Look in GitHub issues, Stack Overflow
4. **Profile**: Use profiling tools to identify bottlenecks
5. **Debug**: Add print statements and logging
6. **Ask**: Check community forums and Discord

---

## References

- [CUDA Debugging Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [Mojo Error Messages](https://docs.modular.com/mojo/manual/decorators)
- [FastAPI Debugging](https://fastapi.tiangolo.com/deployment/concepts/)
- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug-application-cluster/)
