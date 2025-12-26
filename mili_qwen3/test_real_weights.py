#!/usr/bin/env python3
"""
Test script for loading and running Qwen3-0.6B model with real weights.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from python_layer.model.weight_loader import WeightLoader
from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode


def load_qwen3_model():
    """Load Qwen3-0.6B model from local HuggingFace cache."""
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )

    print("Loading Qwen3-0.6B model...")
    print(f"Model path: {model_path}")

    # First, load the config
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import json

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    print("Model config:")
    print(f"  Vocab size: {config_dict.get('vocab_size')}")
    print(f"  Hidden size: {config_dict.get('hidden_size')}")
    print(f"  Num layers: {config_dict.get('num_hidden_layers')}")
    print(f"  Num heads: {config_dict.get('num_attention_heads')}")
    print(f"  Num KV heads: {config_dict.get('num_key_value_heads')}")

    # Create WeightLoader
    weight_loader = WeightLoader(model_path, config_dict, device="cpu", dtype="float32")
    weight_loader.load_from_safetensors()

    print(f"Loaded {len(weight_loader.weights)} weight tensors")

    # Create model config (match real model but smaller for testing)
    test_config = ModelConfig(
        vocab_size=config_dict.get("vocab_size", 151936),
        hidden_size=config_dict.get("hidden_size", 1024),
        num_heads=config_dict.get("num_attention_heads", 16),
        num_kv_heads=config_dict.get("num_key_value_heads", 8),
        head_dim=config_dict.get("hidden_size", 1024)
        // config_dict.get("num_attention_heads", 16),
        intermediate_size=config_dict.get("intermediate_size", 2752),
        num_layers=2,  # Smaller for testing
        max_seq_length=128,
        dtype="float32",
    )

    # Create model with real weights
    model = Qwen3Model(test_config, weight_loader)

    return model


def test_model_generation(model):
    """Test text generation with the loaded model."""
    print("\nTesting model generation...")

    # Simple token IDs (will be wrong, but tests the forward pass)
    input_ids = [1, 2, 3, 4, 5]  # Dummy token IDs

    print(f"Input tokens: {input_ids}")

    try:
        # Forward pass
        logits, kv_cache = model.forward(np.array(input_ids), InferenceMode.PREFILL)
        print(f"Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"KV cache length: {len(kv_cache)}")

        # Generate one token
        next_token_logits = logits[-1]  # Last token logits
        next_token = int(np.argmax(next_token_logits))  # Greedy decoding
        print(f"Predicted next token: {next_token}")

        # Test decode mode
        print("\nTesting decode mode...")
        new_input = [next_token]
        decode_logits, _ = model.forward(np.array(new_input), InferenceMode.DECODE, kv_cache)
        print(f"Decode successful! Logits shape: {decode_logits.shape}")

        return True

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Qwen3-0.6B Real Weights Test")
    print("=" * 60)

    try:
        # Load model
        model = load_qwen3_model()

        # Test generation
        success = test_model_generation(model)

        if success:
            print("\n✅ SUCCESS: Qwen3-0.6B model loaded and working!")
        else:
            print("\n❌ FAILED: Model test failed")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
