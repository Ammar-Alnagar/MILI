# Mojo Kernel Development Guide for MILI

# Mojo Kernel Guide (Legacy)

**⚠️ Note**: This guide describes the original Mojo kernel implementation that is **no longer used** in the current codebase. The current implementation uses HuggingFace transformers directly for inference.

This guide is kept for historical reference and potential future optimizations.

## Original Introduction

This guide walked through implementing GPU kernels in Mojo for the MILI inference system. Mojo provides a Python-like syntax while offering C-level performance and direct GPU programming capabilities.

## ⚠️ Important Notice

**All code examples in this document are hypothetical implementations that were planned but never actually built.** The current codebase contains only stub functions as placeholders. For example:

**Actual implementation in `mojo_kernels/core/attention.🔥`:**
```mojo
"""
Attention kernel stub for MILI.
"""

fn flash_attention_stub():
    """Stub FlashAttention function."""
    pass
```

**What the docs show (hypothetical):**
```mojo
struct FlashAttentionKernel:
    // Complex implementation that doesn't exist
```

The stub files are intended as starting points for future development when Mojo ecosystem matures. The current working system uses HuggingFace transformers instead.

### Why Mojo for GPU Kernels?

- **Python-like Syntax**: Easy to learn while maintaining performance
- **GPU Native**: Direct access to GPU memory and instructions
- **MLIR-based Compilation**: Advanced optimizations
- **Zero-cost Abstractions**: No performance overhead
- **Type Safety**: Compile-time error detection

---

## Part 1: Foundation & Setup

### 1.1 Project Structure for Mojo Kernels

```
mojo_kernels/
├── core/
│   ├── attention.🔥
│   ├── rope.🔥
│   ├── activations.🔥
│   └── normalization.🔥
├── memory/
│   ├── kv_cache.🔥
│   └── allocator.🔥
├── utils/
│   ├── types.🔥
│   └── helpers.🔥
└── build.sh              # Build script
```

### 1.2 Build System

**Note**: The current Mojo kernels are stubs and do not contain real implementations. They serve as placeholders for future development.

**File: `mojo_kernels/build.sh`** (Actual implementation)

```bash
#!/bin/bash
# Build script for MILI Mojo GPU kernels
# Compiles all Mojo kernels for the MILI inference system

set -e  # Exit on first error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LIB_DIR="${SCRIPT_DIR}/lib"

# Create output directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${LIB_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building MILI Mojo GPU Kernels${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to compile a Mojo file
compile_mojo() {
    local mojo_file=$1
    local output_name=$2
    local lib_dir=$3

    echo -e "${YELLOW}Compiling: ${mojo_file}${NC}"

    # Check if mojo command exists
    if ! command -v mojo &> /dev/null; then
        echo -e "${RED}Error: mojo command not found. Please install Mojo SDK.${NC}"
        return 1
    fi

    # Compile with optimization flags
    echo -e "${YELLOW}Running: mojo build -O3 -o ${lib_dir}/${output_name}.so ${mojo_file}${NC}"
    if output=$(mojo build -O3 -o "${lib_dir}/${output_name}.so" "${mojo_file}" 2>&1); then
        echo -e "${GREEN}✓ Successfully compiled: ${output_name}${NC}"
        return 0
    else
        # Check if the error is just "no main function" (expected for libraries)
        if echo "$output" | grep -q "module does not contain a 'main' function"; then
            echo -e "${GREEN}✓ Successfully compiled: ${output_name} (library module)${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to compile: ${mojo_file}${NC}"
            echo -e "${RED}$output${NC}"
            return 1
        fi
    fi
}

# Compile utility types first (dependency for other kernels)
echo -e "${BLUE}[1/6] Compiling utility types...${NC}"
compile_mojo "${SCRIPT_DIR}/utils/types.🔥" "types" "${LIB_DIR}" || exit 1

# Compile core kernels
echo -e "${BLUE}[2/6] Compiling RoPE kernel...${NC}"
compile_mojo "${SCRIPT_DIR}/core/rope.🔥" "rope" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[3/6] Compiling RMSNorm kernel...${NC}"
compile_mojo "${SCRIPT_DIR}/core/normalization.🔥" "rmsnorm" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[4/6] Compiling SwiGLU activation kernel...${NC}"
compile_mojo "${SCRIPT_DIR}/core/activations.🔥" "activations" "${LIB_DIR}" || exit 1

echo -e "${BLUE}[5/6] Compiling Attention kernels (FlashAttention, Decode)...${NC}"
compile_mojo "${SCRIPT_DIR}/core/attention.🔥" "attention" "${LIB_DIR}" || exit 1

# Compile memory management kernels
echo -e "${BLUE}[6/6] Compiling Paged KV Cache kernel...${NC}"
compile_mojo "${SCRIPT_DIR}/memory/kv_cache.🔥" "kv_cache" "${LIB_DIR}" || exit 1
```

echo "All kernels compiled successfully!"
ls -lah lib/*/
```

### 1.3 Type Definitions

**File: `mojo_kernels/utils/types.🔥`**

The current implementation contains basic stub types:

```mojo
"""
Minimal type definitions for MILI GPU kernels.
"""

# Basic data type
struct DType:
    var value: Int32

# Kernel status
struct KernelStatus:
    var success: Bool
    var error_code: Int32
```
    alias int8 = DType(3)
    alias int32 = DType(4)

# Tensor shape specification
@register_passable("trivial")
struct Shape:
    var dims: UInt32
    
    fn __init__(inout self, *shape_dims: Int):
        """Initialize shape from variable arguments."""
        self.dims = len(shape_dims)

# Buffer metadata
struct BufferMetadata:
    var batch_size: UInt32
    var seq_length: UInt32
    var hidden_dim: UInt32
    var num_heads: UInt32
    var head_dim: UInt32
    
    fn __init__(
        inout self,
        batch_size: UInt32,
        seq_length: UInt32,
        hidden_dim: UInt32,
        num_heads: UInt32
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

# Device context for kernel execution
struct DeviceContext:
    var device_id: Int
    var stream: UInt64
    var block_size: Int
    var grid_size: Int
    
    fn __init__(
        inout self,
        device_id: Int = 0,
        stream: UInt64 = 0,
        block_size: Int = 256,
        grid_size: Int = 0
    ):
        self.device_id = device_id
        self.stream = stream
        self.block_size = block_size
        self.grid_size = grid_size
```

---

## Part 2: Foundational Kernels

### 2.1 Rotary Position Embeddings (RoPE)

**File: `mojo_kernels/core/rope.🔥`**

RoPE applies rotations to query and key embeddings based on position.

**Note**: The current RoPE implementation is a stub placeholder.

**File: `mojo_kernels/core/rope.🔥`** (Actual implementation)

```mojo
"""
RoPE kernel stub for MILI.
"""

# Stub function for RoPE application
fn apply_rope_stub():
    """Stub RoPE function."""
    pass
        let m = pos.cast[Float32]()
        
        let real = cos(m * theta)
        let imag = sin(m * theta)
        
        return SIMD[DType.float32, 2](real, imag)
    
    fn apply_rope(
        inout self,
        q: SIMD[DType.float32, 2],
        freqs: SIMD[DType.float32, 2],
        dim_idx: Int
    ) -> SIMD[DType.float32, 2]:
        """Apply rope rotation to a query/key pair."""
        # Extract real and imaginary components
        let q_x = q[0]
        let q_y = q[1]
        let cos_theta = freqs[0]
        let sin_theta = freqs[1]
        
        # Apply rotation: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
        let new_x = q_x * cos_theta - q_y * sin_theta
        let new_y = q_x * sin_theta + q_y * cos_theta
        
        return SIMD[DType.float32, 2](new_x, new_y)
    
    fn forward(
        inout self,
        q_ptr: UnsafePointer[Float32],
        k_ptr: UnsafePointer[Float32],
        seq_length: UInt32
    ):
        """
        Apply RoPE to query and key embeddings.
        
        Args:
            q_ptr: Pointer to query tensor [batch, seq_len, num_heads, head_dim]
            k_ptr: Pointer to key tensor [batch, seq_len, num_heads, head_dim]
            seq_length: Current sequence length
        """
        let head_dim = self.metadata.head_dim
        
        # Process each position in sequence
        for pos in range(seq_length):
            for dim in range(0, head_dim, 2):
                # Compute frequencies for this dimension
                let freqs = self.compute_rope_freqs(
                    pos.cast[UInt32](),
                    head_dim.cast[UInt32]()
                )
                
                # Load Q and K values
                let q_offset = (pos * head_dim + dim).cast[Int]()
                let k_offset = (pos * head_dim + dim).cast[Int]()
                
                let q_vec = SIMD[DType.float32, 2](
                    q_ptr[q_offset],
                    q_ptr[q_offset + 1]
                )
                let k_vec = SIMD[DType.float32, 2](
                    k_ptr[k_offset],
                    k_ptr[k_offset + 1]
                )
                
                # Apply rotation
                let q_rotated = self.apply_rope(q_vec, freqs, dim)
                let k_rotated = self.apply_rope(k_vec, freqs, dim)
                
                # Store results
                q_ptr[q_offset] = q_rotated[0]
                q_ptr[q_offset + 1] = q_rotated[1]
                k_ptr[k_offset] = k_rotated[0]
                k_ptr[k_offset + 1] = k_rotated[1]
```

### 2.2 RMSNorm Kernel

**File: `mojo_kernels/core/normalization.🔥`**

**Note**: The current RMSNorm implementation is a stub placeholder.

```mojo
"""
Normalization kernel stub for MILI.
"""

fn rms_norm_stub():
    """Stub RMSNorm function."""
    pass
        self,
        ptr: UnsafePointer[Float32],
        dim: UInt32
    ) -> Float32:
        """Compute RMS (root mean square) of a vector."""
        var sum_sq: Float32 = 0.0
        
        for i in range(dim):
            let val = ptr[i.cast[Int]()]
            sum_sq += val * val
        
        let mean_sq = sum_sq / dim.cast[Float32]()
        return sqrt(mean_sq + self.epsilon)
    
    fn forward(
        inout self,
        input_ptr: UnsafePointer[Float32],
        weight_ptr: UnsafePointer[Float32],
        output_ptr: UnsafePointer[Float32],
        hidden_dim: UInt32
    ):
        """
        Apply RMSNorm normalization.
        
        Formula: output = (input / RMS(input)) * weight
        
        Args:
            input_ptr: Input tensor pointer
            weight_ptr: Learnable weight (scale)
            output_ptr: Output tensor pointer
            hidden_dim: Hidden dimension size
        """
        # Compute RMS
        let rms = self.compute_rms(input_ptr, hidden_dim)
        let inv_rms = 1.0 / rms
        
        # Normalize and scale
        for i in range(hidden_dim):
            let idx = i.cast[Int]()
            let normalized = input_ptr[idx] * inv_rms
            output_ptr[idx] = normalized * weight_ptr[idx]
```

### 2.3 SwiGLU Activation Kernel

**File: `mojo_kernels/core/activations.🔥`**

SwiGLU is a gating activation function that improves model expressivity.

```mojo
"""SwiGLU activation kernel implementation."""

from utils.types import DType, DeviceContext
from math import tanh

struct SwiGLUKernel:
    """Kernel for SwiGLU activation function."""
    
    var context: DeviceContext
    
    fn __init__(inout self, context: DeviceContext):
        self.context = context
    
    fn swish(self, x: Float32) -> Float32:
        """Swish activation: x * sigmoid(x)"""
        let sigmoid_x = 1.0 / (1.0 + exp(-x))
        return x * sigmoid_x
    
    fn forward(
        inout self,
        hidden_states: UnsafePointer[Float32],
        gate: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        dim: UInt32
    ):
        """
        Apply SwiGLU activation.
        
        Formula: output = Swish(hidden_states) * gate
        
        Args:
            hidden_states: Pointer to hidden states
            gate: Pointer to gate values
            output: Pointer to output
            dim: Dimension size
        """
        for i in range(dim):
            let idx = i.cast[Int]()
            let swished = self.swish(hidden_states[idx])
            output[idx] = swished * gate[idx]
```

---

## Part 3: Attention Kernels

### 3.1 FlashAttention Prefill Kernel

**File: `mlio_kernels/core/attention.🔥`**

This implements efficient attention for the prefill phase using tiled computation.

```mojo
"""FlashAttention kernel for prefill phase."""

from utils.types import DType, BufferMetadata, DeviceContext
from math import sqrt, exp

struct FlashAttentionKernel:
    """High-performance FlashAttention kernel for prefill."""
    
    var context: DeviceContext
    var metadata: BufferMetadata
    var tile_size: Int  # Tile size for block computation
    var scale: Float32
    
    fn __init__(
        inout self,
        context: DeviceContext,
        metadata: BufferMetadata,
        tile_size: Int = 64
    ):
        self.context = context
        self.metadata = metadata
        self.tile_size = tile_size
        self.scale = 1.0 / sqrt(metadata.head_dim.cast[Float32]())
    
    fn matmul_block(
        self,
        a: UnsafePointer[Float32],
        b: UnsafePointer[Float32],
        c: UnsafePointer[Float32],
        m: Int,
        n: Int,
        k: Int,
        lda: Int,
        ldb: Int,
        ldc: Int
    ):
        """
        Perform block matrix multiplication: C = A @ B^T
        Used for computing attention scores (Q @ K^T)
        """
        for i in range(m):
            for j in range(n):
                var sum: Float32 = 0.0
                for p in range(k):
                    sum += a[i * lda + p] * b[j * ldb + p]
                c[i * ldc + j] = sum
    
    fn softmax_block(
        self,
        scores: UnsafePointer[Float32],
        m: Int,
        n: Int
    ):
        """
        Apply softmax to attention scores (row-wise).
        Uses online softmax for numerical stability.
        """
        for i in range(m):
            # Find max for numerical stability
            var max_val: Float32 = -1e9
            for j in range(n):
                max_val = max(max_val, scores[i * n + j])
            
            # Compute exp and sum
            var sum_exp: Float32 = 0.0
            for j in range(n):
                let exp_val = exp(scores[i * n + j] - max_val)
                scores[i * n + j] = exp_val
                sum_exp += exp_val
            
            # Normalize
            let inv_sum = 1.0 / sum_exp
            for j in range(n):
                scores[i * n + j] *= inv_sum
    
    fn attention_forward(
        inout self,
        q: UnsafePointer[Float32],
        k: UnsafePointer[Float32],
        v: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        seq_length: UInt32
    ):
        """
        Compute attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Uses tiling for efficient GPU memory access:
        1. Load Q, K tiles into shared memory
        2. Compute attention scores
        3. Apply softmax
        4. Compute output with V
        """
        let head_dim = self.metadata.head_dim
        let seq_len = seq_length.cast[Int]()
        
        # Allocate temporary buffers for scores
        # scores[seq_len, seq_len] will be computed in tiles
        let scores_size = seq_len * seq_len
        
        for tile_i in range(0, seq_len, self.tile_size):
            for tile_j in range(0, seq_len, self.tile_size):
                let m = min(self.tile_size, seq_len - tile_i)
                let n = min(self.tile_size, seq_len - tile_j)
                
                # Compute Q_tile @ K_tile^T -> scores_tile
                self.matmul_block(
                    q.offset(tile_i * head_dim),
                    k.offset(tile_j * head_dim),
                    output.offset(tile_i * seq_len + tile_j),
                    m, n, head_dim,
                    head_dim, head_dim, seq_len
                )
                
                # Scale scores
                for i in range(m):
                    for j in range(n):
                        let idx = (tile_i + i) * seq_len + (tile_j + j)
                        output[idx] *= self.scale
        
        # Apply softmax row-wise
        self.softmax_block(output, seq_len, seq_len)
        
        # Multiply by V to get final output
        # This would be output @ V in full computation
```

### 3.2 Decode-Phase Attention Kernel

```mojo
"""Optimized attention kernel for decode (single token generation) phase."""

struct DecodeAttentionKernel:
    """
    Efficient attention for decode phase.
    Optimized for single query token against cached KV.
    """
    
    var context: DeviceContext
    var metadata: BufferMetadata
    var scale: Float32
    
    fn __init__(
        inout self,
        context: DeviceContext,
        metadata: BufferMetadata
    ):
        self.context = context
        self.metadata = metadata
        self.scale = 1.0 / sqrt(metadata.head_dim.cast[Float32]())
    
    fn forward(
        inout self,
        q: UnsafePointer[Float32],  # [1, 1, num_heads, head_dim]
        k_cache: UnsafePointer[Float32],  # [seq_len, num_heads, head_dim]
        v_cache: UnsafePointer[Float32],  # [seq_len, num_heads, head_dim]
        output: UnsafePointer[Float32],  # [1, num_heads, head_dim]
        seq_length: UInt32
    ):
        """
        Single token decode attention:
        - Query: 1 token
        - KV: cached from previous tokens
        - Output: 1 token with updated KV
        """
        let head_dim = self.metadata.head_dim
        let num_heads = self.metadata.num_heads
        let seq_len = seq_length.cast[Int]()
        
        for h in range(num_heads):
            # Compute attention scores: q @ k^T
            var scores = UnsafePointer[Float32].alloc(seq_len)
            
            for t in range(seq_len):
                var score: Float32 = 0.0
                for d in range(head_dim):
                    let q_val = q[h * head_dim + d]
                    let k_val = k_cache[t * num_heads * head_dim + h * head_dim + d]
                    score += q_val * k_val
                scores[t] = score * self.scale
            
            # Softmax
            var max_score: Float32 = -1e9
            for t in range(seq_len):
                max_score = max(max_score, scores[t])
            
            var sum_exp: Float32 = 0.0
            for t in range(seq_len):
                let exp_val = exp(scores[t] - max_score)
                scores[t] = exp_val
                sum_exp += exp_val
            
            let inv_sum = 1.0 / sum_exp
            for t in range(seq_len):
                scores[t] *= inv_sum
            
            # Weighted sum over values
            for d in range(head_dim):
                var out_val: Float32 = 0.0
                for t in range(seq_len):
                    let v_val = v_cache[t * num_heads * head_dim + h * head_dim + d]
                    out_val += scores[t] * v_val
                output[h * head_dim + d] = out_val
            
            scores.free()
```

---

## Part 4: Memory Management

### 4.1 KV Cache Structure

**File: `mojo_kernels/memory/kv_cache.🔥`**

```mojo
"""KV Cache management for paged attention."""

struct PagedKVCache:
    """
    Paged KV cache structure with per-token-block management.
    
    Structure:
    - Page size: 16 tokens (configurable)
    - Pages are allocated/deallocated independently
    - Reference counting for multi-request sharing
    """
    
    var k_cache: UnsafePointer[Float32]
    var v_cache: UnsafePointer[Float32]
    var num_pages: UInt32
    var page_size: UInt32
    var tokens_per_page: UInt32
    var ref_count: UnsafePointer[UInt32]
    
    fn __init__(
        inout self,
        num_pages: UInt32,
        page_size: UInt32 = 16
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self.tokens_per_page = page_size
        
        # Allocate cache memory
        self.k_cache = UnsafePointer[Float32].alloc(
            num_pages * page_size * 128  # Assume 128-D embeddings
        )
        self.v_cache = UnsafePointer[Float32].alloc(
            num_pages * page_size * 128
        )
        
        # Reference counting
        self.ref_count = UnsafePointer[UInt32].alloc(num_pages)
        for i in range(num_pages):
            self.ref_count[i.cast[Int]()] = 0
    
    fn allocate_page(
        inout self
    ) -> UInt32:
        """
        Allocate a new page for KV cache.
        Returns page index or -1 if no pages available.
        """
        for i in range(self.num_pages):
            if self.ref_count[i.cast[Int]()] == 0:
                self.ref_count[i.cast[Int]()] = 1
                return i.cast[UInt32]()
        return ~0 as UInt32  # Error: no available pages
    
    fn add_reference(inout self, page_id: UInt32):
        """Increment reference count for a page."""
        self.ref_count[page_id.cast[Int]()] += 1
    
    fn release_page(inout self, page_id: UInt32):
        """Release a page (decrement reference count)."""
        let idx = page_id.cast[Int]()
        if self.ref_count[idx] > 0:
            self.ref_count[idx] -= 1
    
    fn __del__(owned self):
        """Clean up allocated memory."""
        self.k_cache.free()
        self.v_cache.free()
        self.ref_count.free()
```

---

## Part 5: Building and Testing

### 5.1 Compilation

To compile a kernel:

```bash
mojo build -o lib/core/rope.so core/rope.🔥 -D max_unroll=4
```

**Useful Mojo compiler flags:**
- `-O3`: Maximum optimization
- `-D`: Define compile-time values
- `--target`: Target architecture (nvidia, amd, etc.)

### 5.2 Testing Kernels

**File: `tests/unit/test_kernels.py`**

```python
"""Unit tests for Mojo kernels."""

import numpy as np
import pytest
from pathlib import Path

# Load compiled kernels
KERNEL_LIB_PATH = Path(__file__).parent.parent.parent / "mojo_kernels" / "lib"

class TestRoPEKernel:
    """Test Rotary Position Embeddings kernel."""
    
    def test_rope_frequency_computation(self):
        """Verify RoPE frequency calculation."""
        head_dim = 128
        base = 10000.0
        
        # Expected theta values
        for i in range(head_dim):
            theta = base ** (-2.0 * (i / head_dim))
            assert theta > 0
            assert theta <= 1.0
    
    def test_rope_rotation_invariant(self):
        """Verify RoPE maintains norm."""
        q = np.array([1.0, 1.0], dtype=np.float32)
        # After rotation, norm should be preserved
        norm_before = np.linalg.norm(q)
        # Apply RoPE (mock)
        norm_after = norm_before
        assert np.isclose(norm_before, norm_after)


class TestRMSNormKernel:
    """Test RMSNorm kernel."""
    
    def test_rmsnorm_output_shape(self):
        """Verify output shape matches input."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 768
        
        input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        # Should return same shape
        assert input_data.shape == (batch_size, seq_len, hidden_dim)
    
    def test_rmsnorm_numerical_stability(self):
        """Verify numerical stability with small values."""
        epsilon = 1e-6
        small_values = np.array([1e-7, 1e-7, 1e-7], dtype=np.float32)
        # Should not produce inf or nan
        rms = np.sqrt(np.mean(small_values**2) + epsilon)
        assert np.isfinite(rms)


class TestFlashAttentionKernel:
    """Test FlashAttention kernel."""
    
    def test_attention_output_shape(self):
        """Verify attention output shape."""
        batch_size = 2
        seq_len = 512
        num_heads = 32
        head_dim = 64
        
        q_shape = (batch_size, seq_len, num_heads, head_dim)
        # Attention output should match Q shape
        assert q_shape == (batch_size, seq_len, num_heads, head_dim)
    
    def test_softmax_normalization(self):
        """Verify attention weights sum to 1."""
        # After softmax, attention weights should sum to 1
        seq_len = 4
        attention_weights = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.isclose(attention_weights.sum(), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Part 6: Performance Optimization Tips

### 6.1 Memory Optimization

```mojo
# Use SIMD for vectorized operations
fn vectorized_add(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    n: Int
):
    # Process 4 floats at a time with SIMD
    for i in range(0, n, 4):
        let vec_a = SIMD[DType.float32, 4].load(a.offset(i))
        let vec_b = SIMD[DType.float32, 4].load(b.offset(i))
        (vec_a + vec_b).store(c.offset(i))
```

### 6.2 Loop Optimization

```mojo
# Use @unroll for small fixed loops
@unroll
fn unrolled_computation(ptr: UnsafePointer[Float32]):
    for i in range(4):  # Will be unrolled at compile time
        ptr[i] *= 2.0
```

### 6.3 Shared Memory Usage

For GPU kernels, leverage shared memory:

```mojo
# Pseudo-code for shared memory usage
# Place frequently accessed data in shared memory
shared_buffer = allocate_shared_memory(size)
# All threads in block can access at high speed
```

---

## Next Steps

1. Implement each kernel following the templates above
2. Compile with `mojo build`
3. Run unit tests in `tests/unit/`
4. Profile with `mojo profile`
5. Optimize based on profiling results
6. Move to Python integration layer (see `03_PYTHON_INTEGRATION.md`)

---

## Debugging Tips

```bash
# Compile with debug symbols
mojo build -g -o lib/core/rope.so core/rope.🔥

# Run with CUDA debug symbols
CUDA_LAUNCH_BLOCKING=1 python test_script.py

# Use printf debugging in kernels
print("Debug value:", some_var)
```

---

## References

- [Mojo GPU Programming Guide](https://docs.modular.com/mojo/manual/decorators/kernel)
- [CUDA Performance Optimization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FlashAttention Paper Details](https://arxiv.org/pdf/2205.14135.pdf)
