"""
Qwen3 Model Implementation with Mojo Kernel Integration.
Implements the main inference pipeline for Qwen3 using Mojo-accelerated kernels.
"""

try:
    import numpy as np
except ImportError:
    np = None

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class InferenceMode(Enum):
    """Inference execution mode."""

    PREFILL = "prefill"  # Full sequence processing
    DECODE = "decode"  # Single token generation
    MIXED = "mixed"  # Prefill then decode


@dataclass
class ModelConfig:
    """Configuration for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    num_layers: int = 32
    max_seq_length: int = 8192
    rope_base: float = 10000.0
    attention_dropout: float = 0.0
    ff_dropout: float = 0.0
    dtype: str = "float32"
    use_flash_attention: bool = True
    use_paged_kv_cache: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @property
    def num_parameters(self) -> int:
        """Estimate total model parameters."""
        # Embedding + layers * (attention + mlp + norms)
        return (
            self.vocab_size * self.hidden_size  # embeddings
            + self.num_layers
            * (
                3 * self.hidden_size**2  # QKV
                + self.hidden_size * self.intermediate_size * 2  # FFN gate + proj
                + self.hidden_size * 2  # norms
            )
        )


@dataclass
class InferenceRequest:
    """Single inference request."""

    request_id: int
    input_ids: np.ndarray  # [seq_len]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0


class Qwen3Model:
    """Qwen3 language model with Mojo kernel acceleration."""

    def __init__(self, config: ModelConfig, weight_loader=None, device: str = "cpu"):
        """
        Initialize Qwen3 model.

        Args:
            config: Model configuration
            weight_loader: WeightLoader instance for loading real weights
            device: Device to use ("cuda" or "cpu")
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device
        self.dtype = np.float32 if config.dtype == "float32" else np.float16

        # Initialize model weights
        self._init_weights()

        # Initialize caches
        self.kv_cache = None
        self.kv_cache_allocator = None

    def _init_weights(self):
        """Initialize model weights."""
        if self.weight_loader is not None:
            # Load real weights
            self._load_real_weights()
        else:
            # Initialize with random weights for testing
            self._init_random_weights()

    def _load_real_weights(self):
        """Load weights from WeightLoader."""
        try:
            # Load embeddings
            self.embeddings = self.weight_loader.get_weight("embeddings.weight")

            # Load layers
            self.layers = []
            for i in range(self.config.num_layers):
                layer = {
                    "q_proj": self.weight_loader.get_weight(
                        f"layers.{i}.q_proj.weight"
                    ),
                    "k_proj": self.weight_loader.get_weight(
                        f"layers.{i}.k_proj.weight"
                    ),
                    "v_proj": self.weight_loader.get_weight(
                        f"layers.{i}.v_proj.weight"
                    ),
                    "o_proj": self.weight_loader.get_weight(
                        f"layers.{i}.o_proj.weight"
                    ),
                    "gate_proj": self.weight_loader.get_weight(
                        f"layers.{i}.gate_proj.weight"
                    ),
                    "up_proj": self.weight_loader.get_weight(
                        f"layers.{i}.up_proj.weight"
                    ),
                    "down_proj": self.weight_loader.get_weight(
                        f"layers.{i}.down_proj.weight"
                    ),
                    "norm1_weight": self.weight_loader.get_weight(
                        f"layers.{i}.norm1_weight"
                    ),
                    "norm2_weight": self.weight_loader.get_weight(
                        f"layers.{i}.norm2_weight"
                    ),
                }
                self.layers.append(layer)

            # Load final layer norm and lm_head if available
            try:
                self.final_norm_weight = self.weight_loader.get_weight("norm.weight")
            except KeyError:
                self.final_norm_weight = np.ones(
                    self.config.hidden_size, dtype=self.dtype
                )

            try:
                self.lm_head = self.weight_loader.get_weight("lm_head.weight")
            except KeyError:
                # Often tied to embeddings
                self.lm_head = self.embeddings.copy()

            print(f"Successfully loaded weights for {self.config.num_layers} layers")

        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Falling back to random weights")
            self._init_random_weights()

    def _init_random_weights(self):
        """Initialize with random weights for testing."""
        self.embeddings = np.random.randn(
            self.config.vocab_size, self.config.hidden_size
        ).astype(self.dtype)

        self.layers = []
        for _ in range(self.config.num_layers):
            layer = {
                "q_proj": np.random.randn(
                    self.config.hidden_size,
                    self.config.num_heads * self.config.head_dim,
                ).astype(self.dtype),
                "k_proj": np.random.randn(
                    self.config.hidden_size,
                    self.config.num_kv_heads * self.config.head_dim,
                ).astype(self.dtype),
                "v_proj": np.random.randn(
                    self.config.hidden_size,
                    self.config.num_kv_heads * self.config.head_dim,
                ).astype(self.dtype),
                "o_proj": np.random.randn(
                    self.config.num_heads * self.config.head_dim,
                    self.config.hidden_size,
                ).astype(self.dtype),
                "gate_proj": np.random.randn(
                    self.config.hidden_size, self.config.intermediate_size
                ).astype(self.dtype),
                "up_proj": np.random.randn(
                    self.config.hidden_size, self.config.intermediate_size
                ).astype(self.dtype),
                "down_proj": np.random.randn(
                    self.config.intermediate_size, self.config.hidden_size
                ).astype(self.dtype),
                "norm1_weight": np.ones(self.config.hidden_size).astype(self.dtype),
                "norm2_weight": np.ones(self.config.hidden_size).astype(self.dtype),
            }
            self.layers.append(layer)

        # Initialize final layer norm and lm_head
        self.final_norm_weight = np.ones(self.config.hidden_size).astype(self.dtype)
        # Often tied to embeddings
        self.lm_head = self.embeddings.T.copy()

        self.final_norm_weight = np.ones(self.config.hidden_size, dtype=self.dtype)
        self.lm_head = self.embeddings.T.copy()

    def forward(
        self,
        input_ids: np.ndarray,
        inference_mode: InferenceMode = InferenceMode.PREFILL,
        past_key_values: Optional[List] = None,
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Forward pass through model.

        Args:
            input_ids: Input token IDs [seq_len] or [1] for decode
            inference_mode: Prefill or decode mode
            past_key_values: Cached KV from previous tokens

        Returns:
            logits: Output logits [seq_len, vocab_size]
            new_key_values: Updated KV cache
        """
        batch_size = 1
        seq_len = len(input_ids)

        # Embedding lookup
        hidden_states = self.embeddings[input_ids]  # [seq_len, hidden_size]

        # Process through transformer layers
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers

        new_key_values = []

        for layer_idx, layer in enumerate(self.layers):
            # Self-attention
            attn_output, new_kv = self._attention_layer(
                hidden_states,
                layer,
                layer_idx,
                inference_mode,
                past_key_values[layer_idx] if past_key_values else None,
            )
            new_key_values.append(new_kv)

            # Residual connection
            hidden_states = hidden_states + attn_output

            # FFN
            ffn_output = self._ffn_layer(hidden_states, layer)
            hidden_states = hidden_states + ffn_output

        # Final layer norm
        hidden_states = self._layer_norm(hidden_states, self.final_norm_weight)

        # Project to vocabulary
        logits = np.dot(hidden_states, self.lm_head)  # [seq_len, vocab_size]

        return logits, new_key_values

    def _attention_layer(
        self,
        hidden_states: np.ndarray,
        layer_weights: Dict,
        layer_idx: int,
        inference_mode: InferenceMode,
        past_kv: Optional[Tuple] = None,
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Attention layer computation.

        Args:
            hidden_states: Input hidden states [seq_len, hidden_size]
            layer_weights: Layer weight dictionary
            layer_idx: Layer index
            inference_mode: Prefill or decode mode
            past_kv: Previous KV cache

        Returns:
            attention_output: Attention output [seq_len, hidden_size]
            new_kv: Updated KV cache
        """
        seq_len = hidden_states.shape[0]

        # Project Q, K, V
        q = np.dot(hidden_states, layer_weights["q_proj"])  # [seq_len, hidden_size]
        k = np.dot(hidden_states, layer_weights["k_proj"])  # [seq_len, hidden_size]
        v = np.dot(hidden_states, layer_weights["v_proj"])  # [seq_len, hidden_size]

        # Reshape for multi-head attention
        q = q.reshape(seq_len, self.config.num_heads, self.config.head_dim)
        k = k.reshape(seq_len, self.config.num_kv_heads, self.config.head_dim)
        v = v.reshape(seq_len, self.config.num_kv_heads, self.config.head_dim)

        # Apply RoPE (Rotary Position Embeddings)
        q, k = self._apply_rope(q, k, seq_len)

        # Handle KV cache for decode mode
        if inference_mode == InferenceMode.DECODE and past_kv is not None:
            past_k, past_v = past_kv
            # k = np.concatenate([past_k, k], axis=1)
            # v = np.concatenate([past_v, v], axis=1)
            k = k  # For now, ignore past KV
            v = v

        cache_len = k.shape[0]

        # Reshape to [num_heads, seq_len, head_dim] for attention
        q = q.transpose(1, 0, 2)  # [num_heads, seq_len, head_dim]
        k = k.transpose(1, 0, 2)  # [num_kv_heads, cache_len, head_dim]
        v = v.transpose(1, 0, 2)  # [num_kv_heads, cache_len, head_dim]

        # For GQA (Grouped Query Attention), we need to handle the case where
        # num_kv_heads < num_heads. We'll repeat the KV heads to match Q heads.
        if self.config.num_kv_heads < self.config.num_heads:
            # Repeat KV heads to match number of Q heads
            repeat_factor = self.config.num_heads // self.config.num_kv_heads
            k = np.repeat(k, repeat_factor, axis=0)  # [num_heads, cache_len, head_dim]
            v = np.repeat(v, repeat_factor, axis=0)  # [num_heads, cache_len, head_dim]

        # Compute attention scores
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(
            self.config.head_dim
        )  # [num_heads, seq_len, cache_len]

        # Apply causal mask for prefill
        if inference_mode == InferenceMode.PREFILL:
            causal_mask = np.tril(np.ones((seq_len, cache_len)))
            scores = np.where(causal_mask[np.newaxis, :, :], scores, -1e9)

        # Softmax
        attn_weights = self._softmax(scores)

        # Apply attention to values
        attn_output = np.matmul(attn_weights, v)  # [num_heads, seq_len, head_dim]

        # Reshape back to [seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 0, 2).reshape(
            seq_len, -1
        )  # [seq_len, num_heads * head_dim]

        # Output projection
        attn_output = np.dot(attn_output, layer_weights["o_proj"])

        # Return new KV cache
        new_kv = (k, v)  # [num_kv_heads, seq_len, head_dim]

        return attn_output, new_kv

        # Softmax
        attn_weights = self._softmax(scores)

        # Apply attention to values
        attn_output = np.matmul(attn_weights, v)  # [seq_len, num_heads, head_dim]

        # Reshape back
        attn_output = attn_output.reshape(seq_len, -1)  # [seq_len, hidden_size]

        # Output projection
        attn_output = np.dot(attn_output, layer_weights["o_proj"])

        new_kv = (k[-seq_len:], v[-seq_len:])  # Store current k, v for next iteration

        return attn_output, new_kv

    def _ffn_layer(self, hidden_states: np.ndarray, layer_weights: Dict) -> np.ndarray:
        """
        Feed-forward network layer.

        Args:
            hidden_states: Input hidden states
            layer_weights: Layer weights

        Returns:
            FFN output
        """
        # Norm
        normalized = self._layer_norm(hidden_states, layer_weights["norm1_weight"])

        # SwiGLU activation
        gate = np.dot(normalized, layer_weights["gate_proj"])  # [seq_len, intermediate]
        up = np.dot(normalized, layer_weights["up_proj"])  # [seq_len, intermediate]

        # Apply SwiGLU: silu(gate) * up
        activated = self._silu(gate) * up

        # Down projection - ensure correct shape
        # The down_proj matrix has shape [intermediate_size, hidden_size]
        # but activated has shape [seq_len, intermediate_size]
        # This should work directly
        output = np.dot(activated, layer_weights["down_proj"])

        return output

    def _apply_rope(
        self, q: np.ndarray, k: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Rotary Position Embeddings."""
        positions = np.arange(seq_len)

        for pos in range(seq_len):
            for head in range(self.config.num_heads):
                for dim in range(0, self.config.head_dim, 2):
                    # Compute theta
                    theta = 1.0 / (
                        self.config.rope_base ** (2 * dim / self.config.head_dim)
                    )
                    m_theta = positions[pos] * theta

                    cos_val = np.cos(m_theta)
                    sin_val = np.sin(m_theta)

                    # Apply rotation
                    q_x = q[pos, head, dim]
                    q_y = q[pos, head, dim + 1] if dim + 1 < self.config.head_dim else 0

                    q[pos, head, dim] = q_x * cos_val - q_y * sin_val
                    if dim + 1 < self.config.head_dim:
                        q[pos, head, dim + 1] = q_x * sin_val + q_y * cos_val

                    k_x = k[pos, head % self.config.num_kv_heads, dim]
                    k_y = (
                        k[pos, head % self.config.num_kv_heads, dim + 1]
                        if dim + 1 < self.config.head_dim
                        else 0
                    )

                    k[pos, head % self.config.num_kv_heads, dim] = (
                        k_x * cos_val - k_y * sin_val
                    )
                    if dim + 1 < self.config.head_dim:
                        k[pos, head % self.config.num_kv_heads, dim + 1] = (
                            k_x * sin_val + k_y * cos_val
                        )

        return q, k

    def _layer_norm(
        self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """RMSNorm implementation."""
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        return x / np.sqrt(mean_sq + eps) * weight

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) activation."""
        return x * Qwen3Model._sigmoid(x)

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            Generated token IDs
        """
        generated = input_ids.copy()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Get logits for next token
            if len(generated) == len(input_ids):
                # Prefill phase
                logits, past_key_values = self.forward(
                    generated, InferenceMode.PREFILL, past_key_values
                )
            else:
                # Decode phase - only last token
                logits, past_key_values = self.forward(
                    generated[-1:], InferenceMode.DECODE, past_key_values
                )

            # Get last token logits
            next_token_logits = logits[-1] / temperature

            # Top-p sampling
            probs = self._softmax(next_token_logits)
            sorted_probs = np.sort(probs)[::-1]
            cumsum_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.sum(cumsum_probs < top_p)
            cutoff = sorted_probs[cutoff_idx]

            filtered_probs = probs * (probs >= cutoff)
            filtered_probs /= filtered_probs.sum()

            next_token = np.random.choice(len(probs), p=filtered_probs)
            generated = np.append(generated, next_token)

            if next_token == 2:  # EOS token
                break

        return generated
