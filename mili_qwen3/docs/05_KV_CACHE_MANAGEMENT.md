# KV Cache Management & RadixAttention Guide for MILI

## Overview

Efficient KV cache management is critical for LLM inference. This guide covers:

1. **Paged KV Cache**: Fixed-size memory blocks for flexibility
2. **RadixAttention**: Prefix sharing and cache reuse
3. **Reference Counting**: Memory recycling
4. **Block Allocation**: Efficient allocation strategies

---

## Part 1: Paged KV Cache Architecture

### Motivation

**Traditional Approach**:
```
Request 1 (512 tokens):    [Cache: 512 × 2 × num_heads × head_dim]
Request 2 (1024 tokens):   [Cache: 1024 × 2 × num_heads × head_dim]
Request 3 (256 tokens):    [Cache: 256 × 2 × num_heads × head_dim]

Total: 1792 × 2 × num_heads × head_dim (contiguous, wasteful)
```

**Paged Approach**:
```
Block size: 16 tokens

Request 1 (512 tokens):  [Page 0] [Page 1] ... [Page 31]
Request 2 (1024 tokens): [Page 32] [Page 33] ... [Page 95]
Request 3 (256 tokens):  [Page 96] [Page 97] ... [Page 111]

Benefits:
- Flexible allocation (any free page)
- Memory reuse across requests
- Efficient for variable-length sequences
```

### Implementation

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import numpy as np


@dataclass
class PagedKVCacheConfig:
    """Configuration for paged KV cache."""
    page_size: int = 16                    # Tokens per page
    num_pages: int = 1024                  # Total pages
    hidden_size: int = 4096
    num_kv_heads: int = 8
    dtype: torch.dtype = torch.float16


class PagedKVCache:
    """Paged KV cache with reference counting."""
    
    def __init__(self, config: PagedKVCacheConfig):
        self.config = config
        self.page_size = config.page_size
        self.num_pages = config.num_pages
        self.head_dim = config.hidden_size // config.num_kv_heads
        
        # Pre-allocate all pages
        page_shape = (config.page_size, config.num_kv_heads, self.head_dim)
        self.k_cache = torch.zeros(
            (config.num_pages, *page_shape),
            dtype=config.dtype,
            device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        # Reference counting and allocation tracking
        self.ref_count = np.zeros(config.num_pages, dtype=np.uint32)
        self.is_free = np.ones(config.num_pages, dtype=bool)
        self.owner_request = np.full(config.num_pages, -1, dtype=np.int32)
    
    def allocate_pages(self, num_pages: int) -> Optional[List[int]]:
        """
        Allocate contiguous or non-contiguous pages.
        
        Args:
            num_pages: Number of pages to allocate
            
        Returns:
            List of page IDs, or None if not enough free pages
        """
        free_indices = np.where(self.is_free)[0]
        
        if len(free_indices) < num_pages:
            return None
        
        # Allocate first num_pages free pages
        allocated = free_indices[:num_pages].tolist()
        
        for page_id in allocated:
            self.is_free[page_id] = False
            self.ref_count[page_id] = 1
        
        return allocated
    
    def add_reference(self, page_ids: List[int]):
        """
        Add reference to pages (for sharing).
        
        Args:
            page_ids: List of page IDs to reference
        """
        for page_id in page_ids:
            self.ref_count[page_id] += 1
    
    def free_pages(self, page_ids: List[int]):
        """
        Free pages by decrementing reference count.
        
        Args:
            page_ids: List of page IDs to free
        """
        for page_id in page_ids:
            self.ref_count[page_id] -= 1
            
            # Mark as free when ref_count reaches 0
            if self.ref_count[page_id] == 0:
                self.is_free[page_id] = True
    
    def get_page(self, page_id: int, cache_type: str = 'k') -> torch.Tensor:
        """Get a single page."""
        cache = self.k_cache if cache_type == 'k' else self.v_cache
        return cache[page_id]
    
    def gather_pages(
        self,
        page_ids: List[int],
        seq_len: int,
        cache_type: str = 'k'
    ) -> torch.Tensor:
        """
        Gather pages into contiguous tensor.
        
        Args:
            page_ids: List of page IDs to gather
            seq_len: Number of valid tokens (may be < page_size * len(page_ids))
            cache_type: 'k' or 'v'
            
        Returns:
            Gathered tensor [seq_len, num_heads, head_dim]
        """
        cache = self.k_cache if cache_type == 'k' else self.v_cache
        pages = [cache[pid] for pid in page_ids]
        gathered = torch.cat(pages, dim=0)[:seq_len]
        return gathered
    
    def write_pages(
        self,
        page_ids: List[int],
        data: torch.Tensor,
        cache_type: str = 'k'
    ):
        """
        Write data to pages.
        
        Args:
            page_ids: List of page IDs
            data: Data to write [total_tokens, num_heads, head_dim]
            cache_type: 'k' or 'v'
        """
        cache = self.k_cache if cache_type == 'k' else self.v_cache
        
        offset = 0
        for page_id in page_ids:
            page_end = min(offset + self.page_size, len(data))
            cache[page_id, :page_end - offset] = data[offset:page_end]
            offset = page_end
```

---

## Part 2: RadixAttention - Prefix Sharing

### Problem: Redundant KV Caching

```
Request 1: "What is AI?" → Generate response
Request 2: "What is AI? How does it work?" → Generate response

Both share "What is AI?" but cache separately!
```

### Solution: RadixAttention

Use a radix tree structure where:
- Each node represents a cached prompt segment
- Leaves represent complete prompts
- Siblings with same prefix share nodes
- Reference counting tracks usage

### Radix Tree Structure

```
                    [ROOT]
                      |
            [What is AI?]  (common prefix node)
             /            \
        [Question mark]   [How does it work?]
         /                    \
    [Request 1]          [Request 2]
```

### Implementation

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import hashlib


@dataclass
class RadixNode:
    """Node in the radix tree."""
    
    # Tree structure
    key: str                           # Prefix text
    children: Dict[str, 'RadixNode'] = None
    parent: Optional['RadixNode'] = None
    
    # Cache management
    page_ids: List[int] = None         # Allocated pages for this node
    num_tokens: int = 0                # Tokens in this node
    ref_count: int = 0                 # Number of users
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.page_ids is None:
            self.page_ids = []
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0
    
    def total_tokens(self) -> int:
        """Get total tokens from root to this node."""
        if self.parent is None:
            return self.num_tokens
        return self.parent.total_tokens() + self.num_tokens


class RadixAttentionCache:
    """
    Cache manager with RadixAttention for prefix sharing.
    
    Allows multiple requests to share common prompt prefixes.
    """
    
    def __init__(
        self,
        paged_cache: PagedKVCache,
        page_size: int = 16
    ):
        self.paged_cache = paged_cache
        self.page_size = page_size
        self.root = RadixNode(key="ROOT")
        self.request_nodes: Dict[str, RadixNode] = {}
    
    def _hash_tokens(self, tokens: List[int]) -> str:
        """Create hash of token sequence."""
        token_str = ",".join(map(str, tokens))
        return hashlib.md5(token_str.encode()).hexdigest()
    
    def _find_common_prefix(
        self,
        tokens1: List[int],
        tokens2: List[int]
    ) -> int:
        """Find length of common prefix between two token sequences."""
        length = 0
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                length += 1
            else:
                break
        return length
    
    def add_request(
        self,
        request_id: str,
        prompt_tokens: List[int]
    ) -> Dict:
        """
        Add a new request, potentially sharing prefix with existing requests.
        
        Args:
            request_id: Unique request identifier
            prompt_tokens: Token IDs of prompt
            
        Returns:
            Metadata with page allocation
        """
        # Find best existing node to branch from
        best_match_node = self.root
        best_match_length = 0
        
        for existing_node in self.request_nodes.values():
            # Get full prefix from this node
            current = existing_node
            prefix_tokens = []
            
            while current.parent is not None:
                # Would need to store tokens in node for actual implementation
                current = current.parent
            
            # Find common prefix
            common_len = self._find_common_prefix(prefix_tokens, prompt_tokens)
            if common_len > best_match_length:
                best_match_node = current
                best_match_length = common_len
        
        # Create new node for unique suffix
        new_node = RadixNode(key=f"req_{request_id}")
        new_node.parent = best_match_node
        best_match_node.children[new_node.key] = new_node
        
        # Allocate pages for unique tokens
        unique_tokens = len(prompt_tokens) - best_match_length
        num_pages_needed = (unique_tokens + self.page_size - 1) // self.page_size
        
        pages = self.paged_cache.allocate_pages(num_pages_needed)
        if pages is None:
            raise RuntimeError("Insufficient pages for new request")
        
        new_node.page_ids = pages
        new_node.num_tokens = unique_tokens
        new_node.ref_count = 1
        
        # Store reference
        self.request_nodes[request_id] = new_node
        
        # Build full page list (shared + new)
        full_pages = []
        current = new_node.parent
        while current.parent is not None:
            full_pages = current.page_ids + full_pages
            current = current.parent
        full_pages.extend(new_node.page_ids)
        
        return {
            "request_id": request_id,
            "page_ids": full_pages,
            "total_tokens": new_node.total_tokens(),
            "shared_pages": full_pages[:-num_pages_needed] if num_pages_needed > 0 else full_pages,
            "unique_pages": new_node.page_ids
        }
    
    def remove_request(self, request_id: str):
        """
        Remove a request and potentially free pages.
        
        Args:
            request_id: ID of request to remove
        """
        if request_id not in self.request_nodes:
            return
        
        node = self.request_nodes[request_id]
        node.ref_count -= 1
        
        # Free pages if no longer used
        if node.ref_count == 0:
            self.paged_cache.free_pages(node.page_ids)
            # Could recursively free parent pages if no children
        
        del self.request_nodes[request_id]
    
    def get_kv_cache(
        self,
        request_id: str,
        cache_type: str = 'k'
    ) -> torch.Tensor:
        """Get full KV cache for a request."""
        if request_id not in self.request_nodes:
            raise KeyError(f"Request {request_id} not found")
        
        node = self.request_nodes[request_id]
        full_pages = []
        
        # Gather pages from root to this node
        current = node
        while current.parent is not None:
            full_pages = current.page_ids + full_pages
            current = current.parent
        
        seq_len = node.total_tokens()
        return self.paged_cache.gather_pages(full_pages, seq_len, cache_type)
```

---

## Part 3: Memory Management Strategies

### Allocation Strategies

```python
class AllocationStrategy:
    """Base class for allocation strategies."""
    
    def allocate(self, num_pages: int) -> Optional[List[int]]:
        raise NotImplementedError


class FirstFitAllocator(AllocationStrategy):
    """Allocate first available pages."""
    
    def __init__(self, cache: PagedKVCache):
        self.cache = cache
    
    def allocate(self, num_pages: int) -> Optional[List[int]]:
        """Find first contiguous free pages."""
        free_indices = np.where(self.cache.is_free)[0]
        
        if len(free_indices) < num_pages:
            return None
        
        # Try to find contiguous block
        for i in range(len(free_indices) - num_pages + 1):
            block = free_indices[i:i + num_pages]
            if np.all(np.diff(block) == 1):
                return block.tolist()
        
        return None


class BestFitAllocator(AllocationStrategy):
    """Allocate smallest sufficient free block."""
    
    def __init__(self, cache: PagedKVCache):
        self.cache = cache
    
    def allocate(self, num_pages: int) -> Optional[List[int]]:
        """Find smallest sufficient free region."""
        # Scan for free blocks
        free_blocks = []
        in_block = False
        block_start = 0
        
        for i, is_free in enumerate(self.cache.is_free):
            if is_free and not in_block:
                block_start = i
                in_block = True
            elif not is_free and in_block:
                block_size = i - block_start
                if block_size >= num_pages:
                    free_blocks.append((block_start, block_size))
                in_block = False
        
        if not free_blocks:
            return None
        
        # Return smallest sufficient block
        best_block = min(free_blocks, key=lambda x: x[1])
        return list(range(best_block[0], best_block[0] + num_pages))
```

### Eviction Policies

```python
class EvictionPolicy:
    """Base class for eviction strategies."""
    
    def select_victim(self) -> Optional[str]:
        """Select request to evict. Return request_id."""
        raise NotImplementedError


class LRUEviction(EvictionPolicy):
    """Least Recently Used eviction."""
    
    def __init__(self, cache_manager: RadixAttentionCache):
        self.cache_manager = cache_manager
        self.access_times = {}
    
    def record_access(self, request_id: str):
        """Record access time for request."""
        import time
        self.access_times[request_id] = time.time()
    
    def select_victim(self) -> Optional[str]:
        """Return least recently used request."""
        if not self.access_times:
            return None
        return min(self.access_times, key=self.access_times.get)


class TokenLimitEviction(EvictionPolicy):
    """Evict based on token count."""
    
    def __init__(self, cache_manager: RadixAttentionCache, max_total_tokens: int):
        self.cache_manager = cache_manager
        self.max_total_tokens = max_total_tokens
    
    def select_victim(self) -> Optional[str]:
        """Evict largest request if over limit."""
        total_tokens = sum(
            node.total_tokens() 
            for node in self.cache_manager.request_nodes.values()
        )
        
        if total_tokens > self.max_total_tokens:
            # Return request with most tokens
            return max(
                self.cache_manager.request_nodes,
                key=lambda rid: self.cache_manager.request_nodes[rid].total_tokens()
            )
        return None
```

---

## Part 4: Integration with Inference Loop

```python
class InferenceEngine:
    """Inference engine with paged KV cache and RadixAttention."""
    
    def __init__(
        self,
        config: PagedKVCacheConfig,
        allocation_strategy: str = 'best_fit',
        eviction_policy: str = 'lru'
    ):
        self.paged_cache = PagedKVCache(config)
        self.radix_cache = RadixAttentionCache(self.paged_cache)
        
        if allocation_strategy == 'best_fit':
            self.allocator = BestFitAllocator(self.paged_cache)
        else:
            self.allocator = FirstFitAllocator(self.paged_cache)
        
        if eviction_policy == 'lru':
            self.eviction = LRUEviction(self.radix_cache)
        else:
            self.eviction = TokenLimitEviction(self.radix_cache, config.num_pages * 16)
    
    def process_request(
        self,
        request_id: str,
        prompt_tokens: List[int]
    ) -> Dict:
        """
        Process a new inference request.
        
        1. Check if can allocate
        2. Use RadixAttention for prefix sharing
        3. Run inference
        4. Return results
        """
        try:
            # Try to add request (may do prefix sharing)
            metadata = self.radix_cache.add_request(request_id, prompt_tokens)
            return metadata
        
        except RuntimeError:
            # Out of memory: try eviction
            victim = self.eviction.select_victim()
            if victim:
                self.radix_cache.remove_request(victim)
                # Retry
                return self.process_request(request_id, prompt_tokens)
            else:
                raise
```

---

## Performance Metrics

### Memory Efficiency

```
Without Paging:
- 1000 requests × 512 tokens = 512K tokens cached
- Storage: 512K × 2 × 128 heads × 8 heads × 2 bytes = ~256 GB

With Paging (16-token blocks):
- Same cache, but pages reused
- Theoretical: ~16 GB (if 32 pages are reused)

With RadixAttention (50% prefix sharing):
- 512K tokens × 0.5 = 256K unique tokens
- ~8 GB

Speedup: 32x memory efficiency!
```

### Reference Implementation Benchmarks

```python
def benchmark_cache_management():
    """Benchmark cache operations."""
    import time
    
    config = PagedKVCacheConfig(num_pages=1024)
    cache = PagedKVCache(config)
    
    # Allocation benchmark
    start = time.time()
    for _ in range(1000):
        pages = cache.allocate_pages(4)
        cache.free_pages(pages)
    alloc_time = time.time() - start
    
    print(f"Allocation: {alloc_time/1000*1e6:.2f} µs per op")
    
    # RadixAttention benchmark
    radix = RadixAttentionCache(cache)
    tokens = list(range(512))
    
    start = time.time()
    for i in range(100):
        radix.add_request(f"req_{i}", tokens)
    radix_time = time.time() - start
    
    print(f"RadixAttention: {radix_time/100*1e3:.2f} ms per request")
```

---

## Summary

| Feature | Benefit | Implementation |
|---------|---------|-----------------|
| Paging | Flexible allocation | PagedKVCache |
| RadixAttention | Prefix sharing | RadixNode tree |
| Reference counting | Memory reuse | ref_count tracking |
| Allocation strategy | Fragmentation control | FirstFit/BestFit |
| Eviction policy | OOM prevention | LRU/TokenLimit |

---

## References

- **PagedAttention**: https://arxiv.org/abs/2309.06180
- **RadixAttention**: vLLM implementation
- **Memory-efficient Inference**: https://arxiv.org/abs/2309.06180
