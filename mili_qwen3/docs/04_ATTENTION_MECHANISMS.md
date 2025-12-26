# Attention Mechanisms Guide for MILI

## Overview

This guide explains the attention mechanisms used in MILI, focusing on:
1. **Standard Scaled Dot-Product Attention** (baseline)
2. **Grouped Query Attention (GQA)** (Qwen3 specific)
3. **FlashAttention** (optimized prefill)
4. **Decode-Phase Attention** (optimized single-token generation)
5. **Multi-Query Attention** (MQA variant)

---

## Part 1: Scaled Dot-Product Attention

### Mathematical Foundation

The standard attention mechanism:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- **Q (Query)**: [batch_size, seq_len, num_heads, head_dim]
- **K (Key)**: [batch_size, seq_len, num_heads, head_dim]
- **V (Value)**: [batch_size, seq_len, num_heads, head_dim]
- **d_k**: head dimension (typically 64 or 128)

### Computational Steps

1. **Compute Scores**: Q @ K^T → [batch_size, num_heads, seq_len, seq_len]
2. **Scale**: Divide by sqrt(d_k) for stability
3. **Mask**: Apply causal mask for autoregressive generation
4. **Softmax**: Normalize across key dimension
5. **Aggregate**: Apply weights to V values

### Python Reference Implementation

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = True,
    attn_dropout: float = 0.0
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Args:
        q: Query [batch_size, num_heads, seq_len, head_dim]
        k: Key [batch_size, num_heads, seq_len, head_dim]
        v: Value [batch_size, num_heads, seq_len, head_dim]
        causal_mask: Apply causal mask for autoregressive
        attn_dropout: Dropout probability
        
    Returns:
        attention: [batch_size, num_heads, seq_len, head_dim]
    """
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    
    # Apply causal mask
    if causal_mask:
        seq_len = q.shape[-2]
        causal_mask_matrix = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores.masked_fill_(causal_mask_matrix, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if attn_dropout > 0:
        attn_weights = F.dropout(attn_weights, p=attn_dropout, training=True)
    
    # Apply to values
    output = torch.matmul(attn_weights, v)
    
    return output
```

---

## Part 2: Grouped Query Attention (GQA)

### Motivation

**Problem**: Standard multi-head attention requires a KV cache of size `O(seq_len × num_heads × head_dim)`, which is expensive.

**Solution**: Use fewer KV heads, sharing them across query heads.

### Architecture

```
Standard MHA (32 query heads, 32 KV heads):
Query:  [32 heads] ━━━━━━━━━━━━━━━━━━━━━
                     ↓
Key:    [32 heads]
Value:  [32 heads]

Grouped Query Attention (32 query heads, 8 KV heads):
Query:  [32 heads] ━┬─────┬─────┬─────┬━
                    ↓     ↓     ↓     ↓
Key:    [8 heads]  ━┴─────┴─────┴─────┴━
Value:  [8 heads]
```

### Implementation

```python
class GroupedQueryAttention(torch.nn.Module):
    """Grouped Query Attention layer."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        
        # Compute scaling factor for grouped heads
        self.num_query_groups = num_attention_heads // num_key_value_heads
        
        # Linear projections
        self.q_proj = torch.nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache = None
    ) -> torch.Tensor:
        """
        Forward pass with GQA.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Causal mask
            kv_cache: Previous KV cache
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        k = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        
        # Handle KV cache (for incremental generation)
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
        
        # Expand KV for grouped query attention
        # [batch, num_kv_heads, seq_len, head_dim] 
        # -> [batch, num_query_heads, seq_len, head_dim]
        k_expanded = k.repeat_interleave(self.num_query_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_query_groups, dim=1)
        
        # Compute attention
        attn_output = scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            causal_mask=True,
            attn_dropout=self.attention_dropout
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output
```

---

## Part 3: FlashAttention for Prefill

### Key Insight

**Standard Attention IO**: 
- Read Q, K, V: O(seq_len²) data (for computing all pairs)
- Write attention output: O(seq_len) data
- **Total**: O(seq_len²) memory access

**FlashAttention Optimization**:
1. **Tiled Computation**: Process in blocks to fit in fast memory (SRAM)
2. **Online Softmax**: Compute softmax incrementally without materializing full matrix
3. **Efficient Scheduling**: Minimize data movement between GPU memory levels

### Algorithm Overview

```
for block_i in range(0, N, Br):
    for block_j in range(0, N, Bc):
        # Load Q block and KV block into fast memory
        Q_block = load(Q[block_i : block_i+Br])
        KV_block = load(KV[block_j : block_j+Bc])
        
        # Compute scores (stays in fast memory)
        S = Q_block @ KV_block.T
        
        # Compute softmax incrementally
        m, l = online_softmax_update(S, m, l)
        
        # Write O block back to slow memory
        O_block = softmax(S) @ V_block
```

### Complexity Analysis

| Operation | Standard | FlashAttention |
|-----------|----------|----------------|
| FLOPs | O(N d²) | O(N d²) |
| Memory Access | O(N² d) | O(N d²/Br) |
| Speedup | 1x | ~3-4x |

Where N = sequence length, d = head dimension

### Mojo Implementation Sketch

```mojo
struct FlashAttentionConfig:
    var block_size_m: Int = 64   # Q block size
    var block_size_n: Int = 64   # KV block size
    var block_size_k: Int = 64   # Head dimension blocks

fn flash_attention[
    dtype: DType,
    Br: Int = 64,
    Bc: Int = 64
](
    q: Tensor[dtype],      # [N, d]
    k: Tensor[dtype],      # [N, d]
    v: Tensor[dtype],      # [N, d]
) -> Tensor[dtype]:
    """FlashAttention forward pass."""
    
    let N = q.shape[0]
    let d = q.shape[1]
    let O = zeros[dtype](N, d)
    
    for m in range(0, N, Br):
        for n in range(0, N, Bc):
            # Load blocks into shared memory
            let Q_block = load_block(q, m, Br, d)
            let K_block = load_block(k, n, Bc, d)
            let V_block = load_block(v, n, Bc, d)
            
            # Compute attention scores
            let S = matmul(Q_block, K_block.T)
            
            # Apply causal mask if n > m
            if n >= m + Br:
                mask_lower_triangular(S)
            
            # Online softmax
            let P = softmax(S)
            
            # Compute output
            let O_block = matmul(P, V_block)
            store_block(O, O_block, m, Br)
    
    return O
```

---

## Part 4: Decode-Phase Attention

### Problem Context

During generation, we produce one token at a time:

```
Step 1: Input = "Hello"        → Output logits (vocab_size)
                                 Sample: "world"
Step 2: Input = "Hello world"  → Output logits
                                 Sample: "!"
...
```

**Challenge**: Recomputing attention on all previous tokens is wasteful.

**Solution**: Use KV cache to store previous K and V values.

### Optimized Single-Token Attention

```python
def decode_attention(
    q_new: torch.Tensor,      # [batch, num_heads, 1, head_dim]
    k_cache: torch.Tensor,    # [batch, num_heads, seq_len, head_dim]
    v_cache: torch.Tensor,    # [batch, num_heads, seq_len, head_dim]
) -> torch.Tensor:
    """
    Compute attention with cached KV.
    
    - Query: 1 new token
    - Keys/Values: All previous tokens
    - Output: 1 token with attention over all history
    """
    
    # Scores: [batch, num_heads, 1, seq_len]
    scores = torch.matmul(q_new, k_cache.transpose(-2, -1))
    scores = scores / (q_new.shape[-1] ** 0.5)
    
    # Softmax over cached KV
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values: [batch, num_heads, 1, head_dim]
    output = torch.matmul(attn_weights, v_cache)
    
    return output
```

### Performance Considerations

| Metric | Prefill (N tokens) | Decode (1 token) |
|--------|-------------------|-----------------|
| Q × K^T | O(N²) | O(N) |
| Memory Access | O(N²) | O(N) |
| KV Cache | N/A | Cached |

**Key Optimization**: Decode phase is **compute-bound** (matmul-limited) rather than memory-bound.

---

## Part 5: Multi-Request Attention with Paging

### Challenges with Batching

When processing multiple requests:

```
Request 1: seq_len = 512
Request 2: seq_len = 1024
Request 3: seq_len = 256

Problem: Concatenating into single matrix wastes memory for shorter sequences
```

### Ragged/Padded Batching with Paging

```python
class PagedBatchAttention:
    """Attention with paged KV cache for variable-length sequences."""
    
    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self.k_pages = {}
        self.v_pages = {}
    
    def forward(
        self,
        q: torch.Tensor,                    # [total_tokens, num_heads, head_dim]
        page_table: List[List[int]],        # Page assignments per request
        token_offsets: torch.Tensor,        # [batch_size] where each request starts
        request_lengths: torch.Tensor       # [batch_size] length of each request
    ) -> torch.Tensor:
        """
        Compute attention with paged KV cache.
        
        Each request accesses its own pages, allowing sharing of common prefixes.
        """
        
        batch_size = len(page_table)
        output = []
        
        for i in range(batch_size):
            # Get pages for this request
            pages = page_table[i]
            seq_len = request_lengths[i]
            
            # Reconstruct K, V from pages
            k_request = self._gather_pages(self.k_pages, pages, seq_len)
            v_request = self._gather_pages(self.v_pages, pages, seq_len)
            
            # Attention for this request
            q_request = q[token_offsets[i]:token_offsets[i] + 1]  # Single token
            attn = scaled_dot_product_attention(q_request, k_request, v_request)
            
            output.append(attn)
        
        return torch.cat(output, dim=0)
    
    def _gather_pages(self, pages_dict: dict, page_ids: List[int], seq_len: int) -> torch.Tensor:
        """Reconstruct tensor from pages."""
        pages = [pages_dict[pid] for pid in page_ids]
        return torch.cat(pages, dim=0)[:seq_len]
```

---

## Part 6: Grouped Query Attention with Paging

### Integration

Combining GQA with paged KV cache:

```python
def grouped_query_attention_paged(
    q: torch.Tensor,                           # Query
    page_table: List[List[int]],               # Pages per request
    token_offsets: torch.Tensor,               # Request offsets
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int = 16
) -> torch.Tensor:
    """
    GQA with paged KV cache.
    
    Benefits:
    1. GQA reduces KV cache size (fewer heads)
    2. Paging allows efficient sharing across requests
    3. Decode phase only accesses needed pages
    """
    
    # Expand KV heads to match query heads
    num_query_groups = num_query_heads // num_kv_heads
    
    # Process each request
    # ...
```

---

## Benchmarking

### Setup

```python
import torch
import time

def benchmark_attention(seq_len: int, num_heads: int, head_dim: int):
    """Benchmark attention implementations."""
    
    q = torch.randn(1, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(1, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(1, num_heads, seq_len, head_dim, device='cuda')
    
    # Standard attention
    start = time.time()
    for _ in range(100):
        output = scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    print(f"Sequence length: {seq_len}")
    print(f"Standard attention: {std_time/100*1000:.3f} ms")
    print()

benchmark_attention(512, 32, 64)
benchmark_attention(2048, 32, 64)
benchmark_attention(8192, 32, 64)
```

---

## Summary Table

| Variant | Use Case | KV Cache | Speedup |
|---------|----------|----------|---------|
| Standard MHA | Prefill | Full | 1x |
| GQA | All phases | Reduced | 1-2x |
| FlashAttention | Prefill | Full | 3-4x |
| Decode Phase | Generate | Cached | 10-100x |
| Paged GQA | Batched decode | Shared pages | 2-5x |

---

## References

- **FlashAttention**: https://arxiv.org/abs/2205.14135
- **Grouped Query Attention**: https://arxiv.org/abs/2305.13245
- **PagedAttention**: https://arxiv.org/abs/2309.06180
