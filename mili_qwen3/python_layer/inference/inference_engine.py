"""
Inference Engine for Qwen3 with continuous batching and KV cache management.
Orchestrates model execution with Mojo kernel acceleration.
"""

try:
    import numpy as np
except ImportError:
    np = None

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class GenerationPhase(Enum):
    """Phases of token generation."""
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class GenerationOutput:
    """Output from generation step."""
    token_ids: List[int] = field(default_factory=list)
    logits: Optional[np.ndarray] = None
    completion_reason: Optional[str] = None


class InferenceEngine:
    """Main inference engine for Qwen3 model."""
    
    def __init__(
        self,
        model,
        kv_cache,
        batch_size: int = 32,
        max_batch_tokens: int = 4096
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Qwen3Model instance
            kv_cache: PagedKVCache instance
            batch_size: Maximum batch size
            max_batch_tokens: Maximum tokens in batch
        """
        self.model = model
        self.kv_cache = kv_cache
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        
        # Request tracking
        self.active_requests: Dict[int, RequestState] = {}
        self.pending_requests: List[InferenceRequest] = []
        self.completed_requests: Dict[int, GenerationOutput] = {}
    
    def add_request(self, request: 'InferenceRequest'):
        """Add a new request to the engine."""
        self.pending_requests.append(request)
    
    def step(self) -> List[Tuple[int, GenerationOutput]]:
        """
        Execute one inference step.
        
        Returns:
            List of (request_id, output) for completed requests
        """
        # Get next batch
        batch = self._schedule_batch()
        
        if not batch:
            return []
        
        # Execute batch
        outputs = self._execute_batch(batch)
        
        # Post-process results
        completed = []
        for request_id, output in outputs.items():
            request = self.active_requests[request_id]
            
            # Check if generation complete
            if self._is_complete(request, output):
                self.completed_requests[request_id] = output
                del self.active_requests[request_id]
                self.kv_cache.free_pages(request_id)
                completed.append((request_id, output))
            else:
                # Update request state
                request.output_ids.extend(output.token_ids)
                request.generated_tokens += len(output.token_ids)
        
        return completed
    
    def generate(self, prompt_ids: np.ndarray, max_new_tokens: int = 100) -> np.ndarray:
        """
        Generate tokens from prompt.
        
        Args:
            prompt_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated token IDs including prompt
        """
        request = InferenceRequest(
            request_id=0,
            input_ids=prompt_ids.tolist(),
            max_new_tokens=max_new_tokens
        )
        
        self.add_request(request)
        
        # Generate tokens
        generated = prompt_ids.copy()
        
        while len(generated) - len(prompt_ids) < max_new_tokens:
            completed = self.step()
            
            if completed:
                for req_id, output in completed:
                    generated = np.concatenate([
                        generated,
                        np.array(output.token_ids, dtype=prompt_ids.dtype)
                    ])
            
            if not self.active_requests and not self.pending_requests:
                break
        
        return generated
    
    def _schedule_batch(self) -> List['InferenceRequest']:
        """Schedule next batch of requests."""
        batch = []
        total_tokens = 0
        
        # Combine pending and active requests
        candidates = list(self.pending_requests)
        for req_id, state in self.active_requests.items():
            candidates.append(state.request)
        
        for request in candidates:
            if request not in batch:
                # Estimate tokens needed
                tokens_needed = len(request.input_ids) + request.max_new_tokens
                
                if len(batch) < self.batch_size and total_tokens + tokens_needed <= self.max_batch_tokens:
                    batch.append(request)
                    total_tokens += tokens_needed
        
        # Remove processed pending requests
        for req in batch:
            if req in self.pending_requests:
                self.pending_requests.remove(req)
        
        return batch
    
    def _execute_batch(self, batch: List['InferenceRequest']) -> Dict[int, GenerationOutput]:
        """Execute a batch of requests."""
        outputs = {}
        
        for request in batch:
            # Allocate cache if needed
            if request.request_id not in self.active_requests:
                cache_block = self.kv_cache.allocate_pages(
                    request.request_id,
                    (len(request.input_ids) + self.model.config.max_seq_length) // self.kv_cache.page_size
                )
                if not cache_block:
                    continue  # Skip if no memory
                
                self.active_requests[request.request_id] = RequestState(request=request)
            
            state = self.active_requests[request.request_id]
            
            # Determine phase
            if state.generated_tokens == 0:
                phase = GenerationPhase.PREFILL
                input_ids = np.array(request.input_ids, dtype=np.int32)
            else:
                phase = GenerationPhase.DECODE
                input_ids = np.array([state.last_token_id], dtype=np.int32)
            
            # Run model
            logits, kv = self.model.forward(
                input_ids,
                past_key_values=state.past_key_values
            )
            
            state.past_key_values = kv
            state.generated_tokens += 1
            
            # Sample next token
            next_token = self._sample_token(logits[-1], request)
            state.last_token_id = next_token
            
            output = GenerationOutput(token_ids=[next_token], logits=logits)
            outputs[request.request_id] = output
        
        return outputs
    
    def _sample_token(
        self,
        logits: np.ndarray,
        request: 'InferenceRequest'
    ) -> int:
        """Sample next token from logits."""
        # Apply temperature
        logits = logits / request.temperature
        
        # Compute probabilities
        max_logits = np.max(logits)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-k filtering
        top_k_indices = np.argsort(probs)[-request.top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        
        # Top-p filtering
        sorted_indices = np.argsort(top_k_probs)[::-1]
        sorted_probs = top_k_probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        
        filtered_indices = sorted_indices[cumsum < request.top_p]
        filtered_probs = sorted_probs[cumsum < request.top_p]
        filtered_probs /= np.sum(filtered_probs)
        
        # Sample
        final_indices = top_k_indices[filtered_indices]
        token = np.random.choice(final_indices, p=filtered_probs)
        
        return int(token)
    
    def _is_complete(
        self,
        request: 'InferenceRequest',
        output: GenerationOutput
    ) -> bool:
        """Check if generation is complete."""
        # Check token limit
        if request.generated_tokens >= request.max_new_tokens:
            return True
        
        # Check EOS token
        if output.token_ids and output.token_ids[0] == 2:
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        cache_stats = self.kv_cache.get_cache_memory_usage()
        
        return {
            'active_requests': len(self.active_requests),
            'pending_requests': len(self.pending_requests),
            'completed_requests': len(self.completed_requests),
            'cache_memory': cache_stats,
            'throughput_tokens_per_sec': self._estimate_throughput()
        }
    
    def _estimate_throughput(self) -> float:
        """Estimate current throughput."""
        if not self.active_requests:
            return 0.0
        
        # Simple estimate based on active requests
        return len(self.active_requests) * 50.0  # Placeholder


@dataclass
class RequestState:
    """State tracking for an active request."""
    request: 'InferenceRequest'
    generated_tokens: int = 0
    last_token_id: int = 0
    past_key_values: Optional[List] = None
    start_time: float = field(default_factory=time.time)


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: int
    input_ids: List[int]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    output_ids: List[int] = field(default_factory=list)
    generated_tokens: int = 0
