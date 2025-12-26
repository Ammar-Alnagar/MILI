"""
Weight loading and management for MILI Qwen3 inference.
Supports loading from safetensors and HuggingFace models.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import json
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import safetensors.torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

print(f"DEBUG: HAS_TORCH={HAS_TORCH}, HAS_SAFETENSORS={HAS_SAFETENSORS}")


class WeightLoader:
    """Load and manage model weights for Qwen3."""

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        device: str = "cpu",
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize weight loader.

        Args:
            model_path: Path to model directory or HuggingFace model name
            config: Model configuration dict
            device: Device to load weights to ("cpu", "cuda")
            dtype: Data type for weights
        """
        self.model_path = Path(model_path).expanduser()
        self.config = config or {}
        self.device = device
        self.dtype = dtype
        self.weights: Dict[str, np.ndarray] = {}

        # Model architecture info
        self.vocab_size = self.config.get("vocab_size", 151936)
        self.hidden_size = self.config.get("hidden_size", 4096)
        self.num_layers = self.config.get("num_hidden_layers", 32)
        self.num_heads = self.config.get("num_attention_heads", 32)
        self.num_kv_heads = self.config.get("num_key_value_heads", 8)
        self.head_dim = self.config.get("hidden_size", 4096) // self.num_heads
        self.intermediate_size = self.config.get("intermediate_size", 11008)

    def load_from_safetensors(self) -> Dict[str, np.ndarray]:
        """Load weights from safetensors format."""
        if HAS_TORCH:
            # Use safetensors.torch to load (handles bfloat16 better)
            try:
                import safetensors.torch

                safetensors_path = self.model_path / "model.safetensors"
                if not safetensors_path.exists():
                    raise FileNotFoundError(
                        f"Safetensors file not found: {safetensors_path}"
                    )

                tensors = safetensors.torch.load_file(
                    str(safetensors_path), device="cpu"
                )
            except ImportError:
                raise ImportError(
                    "safetensors not available. Install with: pip install safetensors"
                )
        else:
            raise ImportError(
                "Neither torch nor safetensors available. Install with: pip install torch safetensors"
            )

        # Convert to numpy arrays and desired dtype
        self.weights = {}
        for name, tensor in tensors.items():
            # Convert torch tensor to numpy
            if hasattr(tensor, "dtype") and str(tensor.dtype) == "torch.bfloat16":
                # Convert bfloat16 to float32 first
                tensor = tensor.to(torch.float32)

            if hasattr(tensor, "numpy"):
                tensor = tensor.numpy()
            elif hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()

            # Convert to target dtype
            if tensor.dtype != self.dtype:
                tensor = tensor.astype(self.dtype)
            self.weights[name] = tensor

        self._remap_huggingface_names()
        return self.weights

    def load_from_huggingface(self, model_name: str) -> Dict[str, np.ndarray]:
        """Load weights from HuggingFace model hub."""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        try:
            from transformers import AutoModelForCausalLM, AutoConfig

            print(f"Loading model {model_name} from HuggingFace...")

            # Load config first
            if not self.config:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                self.config = config.to_dict()
                self._update_config_from_hf(config)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Load in float32 first
                device_map="cpu",  # Load on CPU
                trust_remote_code=True,
            )

            # Convert to numpy
            self.weights = {}
            for name, param in model.named_parameters():
                tensor = param.detach().cpu().numpy()
                if tensor.dtype != self.dtype:
                    tensor = tensor.astype(self.dtype)
                self.weights[name] = tensor

            self._remap_huggingface_names()
            return self.weights

        except Exception as e:
            raise RuntimeError(f"Error loading from HuggingFace: {e}")

    def _update_config_from_hf(self, config):
        """Update config from HuggingFace config object."""
        self.vocab_size = getattr(config, "vocab_size", 151936)
        self.hidden_size = getattr(config, "hidden_size", 4096)
        self.num_layers = getattr(config, "num_hidden_layers", 32)
        self.num_heads = getattr(config, "num_attention_heads", 32)
        self.num_kv_heads = getattr(config, "num_key_value_heads", 8)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = getattr(config, "intermediate_size", 11008)

    def _remap_huggingface_names(self):
        """Remap HuggingFace parameter names to our naming convention."""
        # Qwen models typically have names like:
        # model.embed_tokens.weight -> embeddings
        # model.layers.0.self_attn.q_proj.weight -> layers.0.q_proj
        # etc.

        remapped = {}

        for name, tensor in self.weights.items():
            # Remove 'model.' prefix if present
            if name.startswith("model."):
                name = name[6:]

            # Handle embedding layer
            if "embed_tokens" in name:
                name = name.replace("embed_tokens", "embeddings")
            elif "lm_head" in name:
                name = name.replace("lm_head", "lm_head")

            # Handle transformer layers
            elif "layers." in name:
                # model.layers.0.self_attn.q_proj.weight -> layers.0.q_proj
                parts = name.split(".")
                if len(parts) >= 4 and parts[2] == "self_attn":
                    layer_idx = parts[1]
                    attn_type = parts[3]
                    if attn_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        name = f"layers.{layer_idx}.{attn_type}"
                    elif attn_type == "rotary_emb":
                        continue  # Skip rotary embeddings, handled separately
                elif len(parts) >= 4 and parts[2] == "mlp":
                    layer_idx = parts[1]
                    mlp_type = parts[3]
                    if mlp_type in ["gate_proj", "up_proj", "down_proj"]:
                        name = f"layers.{layer_idx}.{mlp_type}"
                elif len(parts) >= 3 and parts[2] in [
                    "input_layernorm",
                    "post_attention_layernorm",
                ]:
                    layer_idx = parts[1]
                    norm_type = parts[2]
                    if norm_type == "input_layernorm":
                        name = f"layers.{layer_idx}.norm1_weight"
                    elif norm_type == "post_attention_layernorm":
                        name = f"layers.{layer_idx}.norm2_weight"

            # Handle final layer norm
            elif "norm.weight" in name:
                name = "norm.weight"

            remapped[name] = tensor

        self.weights = remapped

    def get_weight(self, name: str) -> np.ndarray:
        """Get a specific weight tensor."""
        if name not in self.weights:
            # Try some common variations
            variations = [
                name,
                f"model.{name}",
                name.replace("layers.", "model.layers."),
                name.replace("embeddings", "embed_tokens"),
                name.replace("lm_head", "lm_head"),
            ]

            for var in variations:
                if var in self.weights:
                    return self.weights[var]

            available = list(self.weights.keys())[:10]
            raise KeyError(f"Weight '{name}' not found. Available: {available}...")

        return self.weights[name]

    def list_weights(self) -> list:
        """List all available weight names."""
        return list(self.weights.keys())

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
        }

    def save_weights(self, path: str):
        """Save weights to numpy .npz file."""
        np.savez(path, **self.weights)

    def load_weights(self, path: str):
        """Load weights from numpy .npz file."""
        data = np.load(path)
        self.weights = {k: v for k, v in data.items()}
