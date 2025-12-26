#!/usr/bin/env python3
"""
Run the MILI model using Python implementation (no Mojo required).
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "mili_qwen3"))

from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode
from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer

def main():
    print("=" * 60)
    print("MILI Python Model Demo")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = QwenTokenizer()
    print(f"✓ Tokenizer loaded with {tokenizer.vocab_size} tokens")
    
    # Create model configuration - match tokenizer vocab size
    config = ModelConfig(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        num_layers=3,
        vocab_size=tokenizer.vocab_size,  # Match tokenizer
        max_seq_length=64,
        intermediate_size=512
    )
    
    # Initialize model
    model = Qwen3Model(config)
    print(f"✓ Model initialized: {config.num_layers} layers, {config.hidden_size} hidden size")
    
    # Test text generation
    prompt_text = "The quick brown fox"
    print(f"\nPrompt: '{prompt_text}'")
    
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt_text)
    print(f"✓ Tokenized to {len(prompt_tokens)} tokens")
    
    # Generate
    generated_tokens = model.generate(
        np.array(prompt_tokens),
        max_new_tokens=20,
        temperature=0.7
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens.tolist())
    print(f"✓ Generated {len(generated_tokens)} total tokens")
    print(f"\nGenerated text: {generated_text}")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    
    test_input = np.array([1, 2, 3, 4, 5])
    logits, kv_cache = model.forward(test_input, InferenceMode.PREFILL)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  KV cache: {len(kv_cache)} layers")
    
    print("\n" + "=" * 60)
    print("🎉 Python implementation working perfectly!")
    print("=" * 60)

if __name__ == "__main__":
    main()
