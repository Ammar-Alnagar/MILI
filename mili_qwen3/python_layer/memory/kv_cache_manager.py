"""
KV Cache Management for Qwen3 inference with paging and RadixAttention support.
Implements efficient memory allocation and prefix sharing for KV caches.
"""

try:
    import numpy as np
except ImportError:
    np = None

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class AllocationStrategy(Enum):
    """Memory allocation strategy."""

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"


class EvictionPolicy(Enum):
    """Cache eviction policy."""

    LRU = "lru"  # Least Recently Used
    FIFO = "fifo"  # First In First Out
    TOKEN_LIMIT = "token_limit"  # Evict based on token count


@dataclass
class PageMetadata:
    """Metadata for a cache page."""

    page_id: int
    is_free: bool
    ref_count: int
    owner_request_id: int
    token_count: int
    last_access_time: float


@dataclass
class CacheBlock:
    """A block of cached KV for a sequence."""

    block_id: int
    request_id: int
    page_ids: List[int]
    token_count: int
    prefix_hash: Optional[int] = None


class PagedKVCache:
    """Paged KV cache with reference counting and prefix sharing."""

    def __init__(
        self,
        page_size: int = 16,
        num_pages: int = 1024,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        dtype=None,
        enable_prefix_sharing: bool = True,
    ):
        """
        Initialize paged KV cache.

        Args:
            page_size: Number of tokens per page
            num_pages: Total number of pages
            head_dim: Dimension of attention heads
            num_kv_heads: Number of KV heads
            dtype: Data type for cache
            enable_prefix_sharing: Enable RadixAttention prefix sharing
        """
        self.page_size = page_size
        self.num_pages = num_pages
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype if dtype is not None else (np.float32 if np else float)
        self.enable_prefix_sharing = enable_prefix_sharing

        # Memory buffers
        page_bytes = page_size * num_kv_heads * head_dim
        if np:
            self.k_cache = np.zeros((num_pages, page_bytes), dtype=self.dtype)
            self.v_cache = np.zeros((num_pages, page_bytes), dtype=self.dtype)
        else:
            self.k_cache = [[0.0] * page_bytes for _ in range(num_pages)]
            self.v_cache = [[0.0] * page_bytes for _ in range(num_pages)]

        # Page management
        self.page_metadata = [
            PageMetadata(
                page_id=i,
                is_free=True,
                ref_count=0,
                owner_request_id=-1,
                token_count=0,
                last_access_time=0.0,
            )
            for i in range(num_pages)
        ]

        # Request tracking
        self.request_blocks: Dict[int, CacheBlock] = {}
        self._free_pages_list: List[int] = list(range(num_pages))

        # Prefix tree for RadixAttention
        self.prefix_tree: Dict[int, List[int]] = defaultdict(list)  # hash -> page_ids
        self.prefix_hashes: Dict[int, int] = {}  # request_id -> prefix_hash

    @property
    def free_pages(self):
        """Get list of free pages."""
        return self._free_pages_list
    
    def free_pages_for_request(self, request_id: int):
        """Free pages allocated to a request."""
        if request_id not in self.request_blocks:
            return
        
        block = self.request_blocks[request_id]
        
        for page_id in block.page_ids:
            if page_id < 0 or page_id >= len(self.page_metadata):
                continue
            
            self.page_metadata[page_id].ref_count = max(0, self.page_metadata[page_id].ref_count - 1)
            
            if self.page_metadata[page_id].ref_count == 0:
                self.page_metadata[page_id].is_free = True
                self._free_pages_list.append(page_id)
        
        del self.request_blocks[request_id]
    
    def free_pages_method(self, request_id: int):
        """Free pages for request (method version)."""
        self.free_pages_for_request(request_id)
    
    def allocate_pages(
        self,
        request_id: int,
        num_pages: int,
        allocation_strategy: AllocationStrategy = AllocationStrategy.FIRST_FIT,
    ) -> Optional[CacheBlock]:
        """
        Allocate pages for a request.

        Args:
            request_id: ID of the request
            num_pages: Number of pages needed
            allocation_strategy: How to select pages

        Returns:
            CacheBlock with allocated pages, or None if insufficient memory
        """
        if len(self._free_pages_list) < num_pages:
            return None

        allocated_pages = []

        for i in range(num_pages):
            if allocation_strategy == AllocationStrategy.FIRST_FIT:
                page_id = self._free_pages_list.pop(0)
            elif allocation_strategy == AllocationStrategy.BEST_FIT:
                # Best fit not implemented in this simple version
                page_id = self._free_pages_list.pop(0)
            else:
                page_id = self._free_pages_list.pop(0)

            allocated_pages.append(page_id)
            self.page_metadata[page_id].is_free = False
            self.page_metadata[page_id].ref_count = 1
            self.page_metadata[page_id].owner_request_id = request_id

        block = CacheBlock(
            block_id=request_id,
            request_id=request_id,
            page_ids=allocated_pages,
            token_count=0,
        )

        self.request_blocks[request_id] = block
        return block

    def release_request(self, request_id: int):
        """Free pages allocated to a request."""
        if request_id not in self.request_blocks:
            return

        block = self.request_blocks[request_id]

        for page_id in block.page_ids:
            if page_id < 0 or page_id >= self.num_pages:
                continue

            self.page_metadata[page_id].ref_count -= 1

            if self.page_metadata[page_id].ref_count == 0:
                self.page_metadata[page_id].is_free = True
                self._free_pages_list.append(page_id)

        del self.request_blocks[request_id]
        if request_id in self.prefix_hashes:
            del self.prefix_hashes[request_id]

    def write_kv(self, request_id: int, k_data: np.ndarray, v_data: np.ndarray):
        """
        Write KV data to cache pages.

        Args:
            request_id: Request ID
            k_data: Key data [seq_len, num_kv_heads, head_dim]
            v_data: Value data [seq_len, num_kv_heads, head_dim]
        """
        if request_id not in self.request_blocks:
            return

        block = self.request_blocks[request_id]
        seq_len = k_data.shape[0]
        block.token_count = seq_len

        # Flatten KV data
        k_flat = k_data.reshape(seq_len, -1)
        v_flat = v_data.reshape(seq_len, -1)

        write_offset = 0
        for page_id in block.page_ids:
            tokens_to_write = min(self.page_size, seq_len - write_offset)
            if tokens_to_write <= 0:
                break

            start_idx = write_offset
            end_idx = write_offset + tokens_to_write

            self.k_cache[
                page_id, : tokens_to_write * self.num_kv_heads * self.head_dim
            ] = k_flat[start_idx:end_idx].flatten()
            self.v_cache[
                page_id, : tokens_to_write * self.num_kv_heads * self.head_dim
            ] = v_flat[start_idx:end_idx].flatten()

            write_offset += tokens_to_write

    def read_kv(
        self, request_id: int, start_token: int = 0, num_tokens: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read KV data from cache.

        Args:
            request_id: Request ID
            start_token: Starting token position
            num_tokens: Number of tokens to read (None = all)

        Returns:
            Tuple of (k_data, v_data)
        """
        if request_id not in self.request_blocks:
            return np.array([]), np.array([])

        block = self.request_blocks[request_id]

        if num_tokens is None:
            num_tokens = block.token_count - start_token

        k_list = []
        v_list = []

        current_token = 0
        for page_id in block.page_ids:
            tokens_in_page = min(self.page_size, block.token_count - current_token)

            if (
                current_token + tokens_in_page > start_token
                and current_token < start_token + num_tokens
            ):
                page_start = max(0, start_token - current_token)
                page_end = min(tokens_in_page, start_token + num_tokens - current_token)
                page_count = page_end - page_start

                if page_count > 0:
                    start_byte = page_start * self.num_kv_heads * self.head_dim
                    end_byte = page_end * self.num_kv_heads * self.head_dim

                    k_list.append(
                        self.k_cache[page_id, start_byte:end_byte].reshape(
                            page_count, self.num_kv_heads, self.head_dim
                        )
                    )
                    v_list.append(
                        self.v_cache[page_id, start_byte:end_byte].reshape(
                            page_count, self.num_kv_heads, self.head_dim
                        )
                    )

            current_token += tokens_in_page

        if k_list:
            k_data = np.concatenate(k_list, axis=0)
            v_data = np.concatenate(v_list, axis=0)
        else:
            k_data = np.array([], dtype=self.dtype).reshape(
                0, self.num_kv_heads, self.head_dim
            )
            v_data = np.array([], dtype=self.dtype).reshape(
                0, self.num_kv_heads, self.head_dim
            )

        return k_data, v_data

    def get_cache_memory_usage(self) -> Dict[str, int]:
        """Get current cache memory usage statistics."""
        used_pages = sum(1 for m in self.page_metadata if not m.is_free)
        page_bytes = self.page_size * self.num_kv_heads * self.head_dim

        return {
            "used_pages": used_pages,
            "total_pages": self.num_pages,
            "free_pages": len(self._free_pages_list),
            "bytes_per_page": page_bytes,
            "total_used_bytes": used_pages * page_bytes * 2,  # K and V
            "total_capacity_bytes": self.num_pages * page_bytes * 2,
        }


class RadixAttentionCache:
    """
    Radix tree-based KV cache with prefix sharing.
    Enables multiple requests to share common prefixes.
    """

    def __init__(self, paged_cache: PagedKVCache, enable_prefix_sharing: bool = True):
        """
        Initialize RadixAttention cache.

        Args:
            paged_cache: Underlying paged KV cache
            enable_prefix_sharing: Enable prefix sharing
        """
        self.paged_cache = paged_cache
        self.enable_prefix_sharing = enable_prefix_sharing

        # Request to prefix mapping
        self.request_prefixes: Dict[int, List[int]] = {}
        self.prefix_counts: Dict[int, int] = defaultdict(int)

    def allocate_with_prefix_sharing(
        self,
        request_id: int,
        prompt_tokens: List[int],
        existing_requests: Optional[Dict[int, List[int]]] = None,
    ) -> CacheBlock:
        """
        Allocate cache with prefix sharing.

        Args:
            request_id: Request ID
            prompt_tokens: Prompt token IDs
            existing_requests: Map of existing request_id -> tokens

        Returns:
            CacheBlock with allocation
        """
        if not self.enable_prefix_sharing or not existing_requests:
            # No prefix sharing, allocate normally
            num_pages_needed = (
                len(prompt_tokens) + self.paged_cache.page_size - 1
            ) // self.paged_cache.page_size
            return self.paged_cache.allocate_pages(request_id, num_pages_needed)

        # Find longest matching prefix
        best_match_request = None
        best_match_length = 0

        for existing_id, existing_tokens in existing_requests.items():
            match_length = 0
            for i in range(min(len(prompt_tokens), len(existing_tokens))):
                if prompt_tokens[i] == existing_tokens[i]:
                    match_length += 1
                else:
                    break

            if match_length > best_match_length:
                best_match_request = existing_id
                best_match_length = match_length

        # Allocate pages for unique suffix
        unique_tokens = len(prompt_tokens) - best_match_length
        num_pages_needed = (
            unique_tokens + self.paged_cache.page_size - 1
        ) // self.paged_cache.page_size

        block = self.paged_cache.allocate_pages(request_id, num_pages_needed)

        # Store prefix info
        if best_match_request is not None:
            self.request_prefixes[request_id] = prompt_tokens[:best_match_length]

        return block

    def release_request(self, request_id: int):
        """Release request from cache."""
        self.paged_cache.release_request(request_id)
        if request_id in self.request_prefixes:
            del self.request_prefixes[request_id]


class FirstFitAllocator:
    """First-fit memory allocation strategy."""

    def __init__(self, cache: PagedKVCache):
        """Initialize allocator."""
        self.cache = cache

    def allocate(self, request_id: int, num_pages: int) -> Optional[CacheBlock]:
        """Allocate pages using first-fit strategy."""
        return self.cache.allocate_pages(
            request_id, num_pages, AllocationStrategy.FIRST_FIT
        )


class TokenLimitEviction:
    """Token limit-based eviction policy."""

    def __init__(self, cache: PagedKVCache, max_tokens: int):
        """
        Initialize eviction policy.

        Args:
            cache: Paged KV cache
            max_tokens: Maximum tokens to cache
        """
        self.cache = cache
        self.max_tokens = max_tokens
        self.token_counts: Dict[int, int] = {}

    def should_evict(self) -> bool:
        """Check if eviction is needed."""
        total_tokens = sum(self.token_counts.values())
        return total_tokens >= self.max_tokens

    def evict_lru_request(self):
        """Evict least recently used request."""
        if not self.token_counts:
            return

        # Find request with least tokens (simplified LRU)
        lru_request = min(self.token_counts.items(), key=lambda x: x[1])[0]
        self.cache.release_request(lru_request)
        del self.token_counts[lru_request]


class ContinuousBatchingScheduler:
    """Scheduler for continuous batching of inference requests."""

    def __init__(
        self,
        kv_cache: PagedKVCache,
        batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
    ):
        """
        Initialize scheduler.

        Args:
            kv_cache: Paged KV cache
            batch_size: Maximum batch size
            max_tokens_per_batch: Maximum tokens in batch
        """
        self.kv_cache = kv_cache
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch

        self.pending_requests: List[InferenceRequest] = []
        self.active_requests: Dict[int, InferenceRequest] = {}

    def schedule(self, request: "InferenceRequest"):
        """Add request to scheduler."""
        self.pending_requests.append(request)

    def get_next_batch(self) -> List["InferenceRequest"]:
        """Get next batch of requests to process."""
        batch = []
        total_tokens = 0

        for i, request in enumerate(self.pending_requests):
            # Estimate tokens for this request
            estimated_tokens = len(request.input_ids) + request.max_new_tokens

            if (
                len(batch) < self.batch_size
                and total_tokens + estimated_tokens <= self.max_tokens_per_batch
            ):
                batch.append(request)
                total_tokens += estimated_tokens
                self.active_requests[request.request_id] = request
            else:
                break

        # Remove processed requests
        self.pending_requests = self.pending_requests[len(batch) :]

        return batch


@dataclass
class InferenceRequest:
    """Single inference request."""

    request_id: int
    input_ids: List[int]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
