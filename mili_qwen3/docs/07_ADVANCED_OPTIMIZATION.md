# Advanced Optimization & Performance Tuning Guide for MILI

## Overview

This guide covers advanced optimization techniques to maximize inference performance and minimize latency for production MILI systems.

---

## Part 1: GPU Memory Optimization

### 1.1 Memory Layout and Access Patterns

**Cache Hierarchy**:
```
L1 Cache (per thread)    : 48-96 KB, very fast
L2 Cache (per SM)        : 128-512 KB, fast
Global Memory            : GB scale, slower
Host Memory              : GB+ scale, slowest
```

### 1.2 Coalesced Memory Access

```python
"""Demonstrating coalesced vs. non-coalesced memory access."""

# ❌ BAD: Non-coalesced access (strided)
def non_coalesced_read(data):
    """Threads read non-contiguous memory."""
    stride = 32
    values = []
    for i in range(1000):
        values.append(data[i * stride])  # Each thread accesses different bank
    return values

# ✅ GOOD: Coalesced access (contiguous)
def coalesced_read(data):
    """Threads read contiguous memory."""
    values = []
    for i in range(1000):
        values.append(data[i])  # All threads access sequential addresses
    return values
```

### 1.3 Memory Pooling and Pre-allocation

```python
"""GPU memory pooling strategy."""

class GPUMemoryPool:
    """Pre-allocate memory to reduce allocation overhead."""
    
    def __init__(self, total_size_gb: int = 20, device: str = "cuda"):
        self.total_size = total_size_gb * 1024 * 1024 * 1024
        self.device = device
        self.pool = torch.empty(self.total_size // 4, dtype=torch.float32, device=device)
        self.free_offset = 0
    
    def allocate(self, size: int) -> torch.Tensor:
        """Allocate from pool (zero-copy if possible)."""
        if self.free_offset + size > self.total_size:
            raise MemoryError("Pool exhausted")
        
        tensor = self.pool[self.free_offset:self.free_offset + size]
        self.free_offset += size
        return tensor
    
    def reset(self):
        """Reset pool for next batch."""
        self.free_offset = 0


# Usage
pool = GPUMemoryPool(total_size_gb=20)
kv_cache = pool.allocate(1024 * 1024 * 1024)  # 1GB for KV cache
```

### 1.4 Quantization for Memory Savings

```python
"""Model quantization strategies."""

class QuantizationStrategy:
    """Different quantization approaches."""
    
    @staticmethod
    def int8_quantization(weights: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Convert to int8 (8x memory savings)."""
        max_val = weights.abs().max()
        scale_factor = 127.0 / max_val
        quantized = (weights * scale_factor).to(torch.int8)
        return quantized
    
    @staticmethod
    def fp8_quantization(weights: torch.Tensor) -> torch.Tensor:
        """Custom fp8 format (4x memory savings)."""
        # Pseudocode - actual implementation requires special CUDA kernels
        pass
    
    @staticmethod
    def int4_quantization(weights: torch.Tensor) -> torch.Tensor:
        """Ultra-compressed int4 (16x memory savings)."""
        # Requires special packing and unpacking kernels
        pass
    
    @staticmethod
    def kv_cache_quantization(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV cache specifically."""
        if bits == 8:
            return QuantizationStrategy.int8_quantization(k_cache), \
                   QuantizationStrategy.int8_quantization(v_cache)
        else:
            raise NotImplementedError(f"{bits}-bit quantization")

# Memory savings calculation
print("Memory Savings:")
print(f"FP16: 1x (baseline)")
print(f"INT8: 2x compression, 50% memory saved")
print(f"FP8: 2x compression, 50% memory saved")
print(f"INT4: 4x compression, 75% memory saved")
```

---

## Part 2: Compute Optimization

### 2.1 Kernel Fusion

```python
"""Fused kernel operations to reduce memory bandwidth."""

class FusedOperations:
    """Combine multiple operations to reduce round trips to global memory."""
    
    @staticmethod
    def fused_rope_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fused: RoPE + Attention + Output
        
        Without fusion:
        1. Read Q from global memory
        2. Apply RoPE (write to global memory)
        3. Read Q from global memory again
        4. Compute attention (write to global memory)
        5. Read attention output from global memory
        6. Compute output (write to global memory)
        
        With fusion: All 6 steps in one kernel pass
        """
        # Pseudocode
        # Load Q, K, V into shared memory (once)
        # Apply RoPE (stays in shared memory)
        # Compute attention scores (stays in shared memory)
        # Apply softmax (stays in shared memory)
        # Multiply by V (stays in shared memory)
        # Write output (once)
        pass
    
    @staticmethod
    def fused_mlp_activation(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: str = "silu"
    ) -> torch.Tensor:
        """
        Fused: Linear + Activation + Linear
        
        Reduces memory reads/writes for MLP layers
        """
        # Single fused kernel instead of 3 separate operations
        pass


# Fusion benefits
print("Fusion Speedups:")
print("RoPE + Attention: 1.3-1.5x speedup")
print("Linear + Activation: 1.2-1.4x speedup")
print("MLP (Linear + Activation + Linear): 1.5-2.0x speedup")
```

### 2.2 Batching Optimization

```python
"""Advanced batching strategies."""

class AdvancedBatchOptimizer:
    """Optimize batching for maximum GPU utilization."""
    
    def __init__(self, max_tokens: int = 65536, target_util: float = 0.9):
        self.max_tokens = max_tokens
        self.target_util = target_util
    
    def compute_optimal_batch_composition(
        self,
        requests: List[Dict]
    ) -> List[List[Dict]]:
        """
        Arrange requests to maximize GPU utilization.
        
        Strategy: Pack requests with varying lengths together
        """
        prefill_batch = []
        decode_batch = []
        
        prefill_tokens = 0
        decode_tokens = 0
        
        for req in requests:
            if req['phase'] == 'prefill':
                if prefill_tokens + len(req['tokens']) <= self.max_tokens:
                    prefill_batch.append(req)
                    prefill_tokens += len(req['tokens'])
            else:  # decode
                if decode_tokens + 1 <= self.max_tokens:
                    decode_batch.append(req)
                    decode_tokens += 1
        
        return [prefill_batch, decode_batch]
    
    def compute_itc_efficiency(self, batch_size: int, seq_len: int) -> float:
        """
        Compute Intensity-to-Cache efficiency.
        
        Higher is better (more compute, less memory bandwidth)
        
        I = (FLOPs per token) / (Bytes per token)
        """
        # Attention FLOPs: 4 * seq_len * head_dim^2
        flops = 4 * seq_len * 64 * 64
        
        # Memory bytes: Q, K, V, Output
        bytes_accessed = 4 * seq_len * 64 * 2  # float16
        
        intensity = flops / bytes_accessed
        return intensity


# Batching efficiency
print("GPU Utilization with Smart Batching:")
print("Random batching: 60-70%")
print("Smart batching (mixed lengths): 85-95%")
```

### 2.3 Attention Optimization

```python
"""Advanced attention optimization techniques."""

class AttentionOptimization:
    """Different attention optimization strategies."""
    
    @staticmethod
    def sliding_window_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int = 4096
    ) -> torch.Tensor:
        """
        Only attend to recent tokens (sliding window).
        
        Reduces attention complexity from O(N²) to O(N*W)
        where W is window size.
        """
        seq_len = q.shape[-2]
        
        # For each query position, only look back window_size tokens
        output = []
        for i in range(seq_len):
            start = max(0, i - window_size)
            
            q_i = q[:, :, i:i+1, :]  # [batch, heads, 1, dim]
            k_window = k[:, :, start:i+1, :]  # [batch, heads, window, dim]
            v_window = v[:, :, start:i+1, :]
            
            scores = torch.matmul(q_i, k_window.transpose(-2, -1))
            scores = scores / (q.shape[-1] ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_window)
            
            output.append(out)
        
        return torch.cat(output, dim=2)
    
    @staticmethod
    def sparse_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparsity_pattern: str = "strided"
    ) -> torch.Tensor:
        """
        Use sparse attention patterns.
        
        Patterns:
        - Strided: Attend to every k-th token
        - Local: Attend to local window + random
        - BigBird: Local + random + global tokens
        """
        # Pseudocode
        pass
    
    @staticmethod
    def multi_query_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_groups: int = 8
    ) -> torch.Tensor:
        """
        Multi-Query Attention: Share KV across query groups.
        
        Reduces KV cache by num_groups factor.
        """
        # All queries in a group share same K, V
        pass


print("Attention Optimization Benefits:")
print("Sliding Window (4K tokens): 2-4x speedup")
print("Sparse Attention: 3-8x speedup")
print("Multi-Query Attention: 2-3x cache reduction")
```

---

## Part 3: Latency Optimization

### 3.1 Pipeline Parallelism

```python
"""Pipeline parallelism for multi-GPU systems."""

class PipelineParallelStage:
    """Single stage in pipeline parallel execution."""
    
    def __init__(
        self,
        layers: List[torch.nn.Module],
        device_id: int,
        num_stages: int
    ):
        self.layers = layers
        self.device_id = device_id
        self.num_stages = num_stages
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute stage forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineParallelExecutor:
    """Execute model in pipeline parallel fashion."""
    
    def __init__(self, stages: List[PipelineParallelStage], batch_size: int = 4):
        self.stages = stages
        self.batch_size = batch_size
        self.num_stages = len(stages)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pipeline parallel forward pass.
        
        Timeline (4 stages, 4 micro-batches):
        GPU0: MB1 -> MB2 -> MB3 -> MB4
        GPU1:      MB1 -> MB2 -> MB3 -> MB4
        GPU2:           MB1 -> MB2 -> MB3 -> MB4
        GPU3:                MB1 -> MB2 -> MB3 -> MB4
        
        All GPUs utilized simultaneously (good scaling)
        """
        # Split into micro-batches
        micro_batch_size = self.batch_size // self.num_stages
        micro_batches = x.split(micro_batch_size)
        
        outputs = []
        activations = []
        
        for mb in micro_batches:
            mb = mb.to(self.stages[0].device_id)
            
            for stage_idx, stage in enumerate(self.stages):
                mb = stage.forward(mb)
                if stage_idx < len(self.stages) - 1:
                    activations.append(mb)
                    mb = mb.to(self.stages[stage_idx + 1].device_id)
            
            outputs.append(mb)
        
        return torch.cat(outputs)


# Scaling efficiency
print("Pipeline Parallel Scaling:")
print("2 GPUs (2 stages): 1.8x speedup (90% efficiency)")
print("4 GPUs (4 stages): 3.2x speedup (80% efficiency)")
print("8 GPUs (8 stages): 5.5x speedup (69% efficiency)")
```

### 3.2 Tensor Parallelism

```python
"""Tensor parallelism for large models."""

class TensorParallelLinear(torch.nn.Module):
    """Linear layer with tensor parallelism."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_partitions: int,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        
        # Split output features across GPUs
        out_per_partition = out_features // num_partitions
        self.weight = torch.nn.Parameter(
            torch.randn(out_per_partition, in_features, device=device)
        )
        self.bias = torch.nn.Parameter(
            torch.randn(out_per_partition, device=device)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y = X @ W^T + b where W is split across GPUs.
        
        Communication: All-gather to collect outputs
        """
        # Local computation
        output = torch.matmul(x, self.weight.T) + self.bias
        
        # All-gather to combine outputs from all GPUs
        # (Simplified - actual impl uses torch.distributed)
        
        return output


print("Tensor Parallel Scaling:")
print("2x parallelism: 1.7x speedup (85% efficiency)")
print("4x parallelism: 3.0x speedup (75% efficiency)")
```

### 3.3 Speculative Decoding

```python
"""Speculative decoding to reduce latency."""

class SpeculativeDecoder:
    """
    Use smaller model to predict multiple tokens.
    Verify with larger model.
    
    If prediction correct: Accept tokens (N speedup)
    If prediction wrong: Recompute with large model (fallback)
    """
    
    def __init__(
        self,
        large_model: torch.nn.Module,
        small_model: torch.nn.Module,
        gamma: int = 4  # Number of speculative tokens
    ):
        self.large_model = large_model
        self.small_model = small_model
        self.gamma = gamma
    
    def speculative_forward(
        self,
        input_ids: torch.Tensor,
        max_tokens: int
    ) -> torch.Tensor:
        """
        Generate with speculation.
        
        1. Small model generates gamma tokens
        2. Large model verifies all gamma tokens at once
        3. Accept correct prefix, reject rest
        4. Repeat
        """
        generated = input_ids.clone()
        
        for step in range(max_tokens // self.gamma):
            # Small model predicts gamma tokens
            spec_tokens = []
            x = generated
            
            for _ in range(self.gamma):
                logits = self.small_model(x[:, -1:])
                token = torch.argmax(logits, dim=-1)
                spec_tokens.append(token)
                x = torch.cat([x, token], dim=1)
            
            # Large model verifies
            x_candidate = torch.cat([generated, torch.cat(spec_tokens, dim=1)], dim=1)
            large_logits = self.large_model(x_candidate)
            
            # Verify each predicted token
            correct_tokens = 0
            for i, spec_token in enumerate(spec_tokens):
                pos = generated.shape[1] + i
                large_pred = torch.argmax(large_logits[:, pos, :], dim=-1)
                
                if large_pred == spec_token:
                    correct_tokens += 1
                else:
                    break
            
            # Accept correct tokens
            if correct_tokens > 0:
                new_tokens = torch.cat(spec_tokens[:correct_tokens], dim=1)
                generated = torch.cat([generated, new_tokens], dim=1)
            else:
                # Fallback: single token from large model
                logits = self.large_model(generated)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, token], dim=1)
        
        return generated


print("Speculative Decoding Benefits:")
print("Gamma=1: 1.0x (baseline)")
print("Gamma=4: 2.5-3.0x speedup")
print("Gamma=8: 3.5-4.5x speedup")
```

---

## Part 4: Profiling and Monitoring

### 4.1 CUDA Profiling

```python
"""Profile CUDA kernels for bottlenecks."""

import torch.profiler as profiler

def profile_inference(model, input_ids: torch.Tensor):
    """Profile inference execution."""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        model(input_ids)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export for visualization
    prof.export_chrome_trace("trace.json")
    print("Profiling data saved to trace.json")
    print("Open in chrome://tracing for visualization")


# Example usage
# model = Qwen3Model(config, weight_loader)
# input_ids = torch.randint(0, 150000, (1, 512)).cuda()
# profile_inference(model, input_ids)
```

### 4.2 Performance Metrics

```python
"""Key performance metrics to track."""

class PerformanceMetrics:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_inference(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        latency_ms: float,
        memory_used_mb: float
    ):
        """Record inference metrics."""
        
        # Throughput metrics
        total_tokens = prompt_tokens + generated_tokens
        throughput = total_tokens / (latency_ms / 1000)  # tokens/sec
        
        # Efficiency metrics
        prefill_rate = prompt_tokens / (latency_ms / 1000)
        decode_rate = generated_tokens / (latency_ms / 1000)
        
        self.metrics[f"throughput"] = throughput
        self.metrics[f"prefill_rate"] = prefill_rate
        self.metrics[f"decode_rate"] = decode_rate
        self.metrics[f"latency_ms"] = latency_ms
        self.metrics[f"memory_mb"] = memory_used_mb
        self.metrics[f"memory_per_token"] = memory_used_mb / total_tokens
    
    def print_summary(self):
        """Print metrics summary."""
        print("\n=== Performance Metrics ===")
        for name, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{name:20s}: {value:10.2f}")
            else:
                print(f"{name:20s}: {value}")


# Key metrics to optimize
print("Target Metrics:")
print("  Throughput: > 100K tokens/sec (prefill)")
print("  Decode rate: > 50 tokens/sec (single)")
print("  Latency: < 100ms (per token)")
print("  Memory: < 50MB per request")
```

---

## Part 5: Production Optimization Checklist

```markdown
## Pre-Deployment Optimization Checklist

### Code Optimization
- [ ] Use kernel fusion for attention operations
- [ ] Enable memory pooling for allocations
- [ ] Implement quantization (int8 or int4)
- [ ] Use sliding window attention if applicable
- [ ] Enable coalesced memory access

### Model Optimization
- [ ] Quantize model weights to int8/int4
- [ ] Enable half-precision (float16) where possible
- [ ] Use grouped query attention
- [ ] Profile all layers for bottlenecks
- [ ] Remove unused parameters

### System Optimization
- [ ] Enable tensor parallelism for large models
- [ ] Implement pipeline parallelism (if multi-GPU)
- [ ] Enable speculative decoding
- [ ] Optimize batching strategy
- [ ] Pre-allocate GPU memory

### Deployment Optimization
- [ ] Set CUDA environment variables
- [ ] Use optimized CUDA libraries (cuBLAS, cuDNN)
- [ ] Enable NVIDIA GPU clock boost
- [ ] Pin GPU to process
- [ ] Use NUMA binding (multi-socket CPUs)

### Monitoring
- [ ] Track GPU utilization
- [ ] Monitor memory usage
- [ ] Log inference latency
- [ ] Alert on performance regressions
- [ ] Profile periodically

### Testing
- [ ] Benchmark against baselines
- [ ] Test with various batch sizes
- [ ] Test with various sequence lengths
- [ ] Load test with concurrent requests
- [ ] Stress test for memory leaks
```

---

## Performance Optimization Summary

| Optimization | Speedup | Memory | Complexity |
|--------------|---------|--------|------------|
| Kernel Fusion | 1.3-2.0x | - | Medium |
| Quantization | - | 4-8x | Medium |
| Sliding Window | 2-4x | - | Medium |
| Tensor Parallelism | 3-4x | - | High |
| Speculative Decoding | 2.5-4x | - | High |
| Memory Pooling | 1.2-1.5x | N/A | Low |
| Batch Optimization | 1.5-2x | - | Medium |

---

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Tensor Parallelism Paper](https://arxiv.org/abs/2104.04473)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [vLLM Optimization](https://github.com/lm-sys/vllm)
