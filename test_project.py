#!/usr/bin/env python3
"""
Simple test script to verify the MILI project is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent / "mili_qwen3"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode
        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    try:
        from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer, MessageFormatter
        print("✓ Tokenizer imports successful")
    except ImportError as e:
        print(f"✗ Tokenizer import failed: {e}")
        return False
    
    try:
        from python_layer.memory.kv_cache_manager import PagedKVCache, RadixAttentionCache
        print("✓ Memory imports successful")
    except ImportError as e:
        print(f"✗ Memory import failed: {e}")
        return False
    
    try:
        from python_layer.inference.inference_engine import InferenceEngine
        print("✓ Inference imports successful")
    except ImportError as e:
        print(f"✗ Inference import failed: {e}")
        return False
    
    return True

def test_tokenizer():
    """Test basic tokenizer functionality."""
    print("\nTesting tokenizer...")
    
    try:
        from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer
        
        tokenizer = QwenTokenizer()
        
        # Test encoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        print(f"✓ Encoded '{text}' to {len(tokens)} tokens")
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        print(f"✓ Decoded back to: '{decoded}'")
        
        # Test special tokens
        assert tokenizer.vocab_size > 0
        print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
        
        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False

def test_model():
    """Test basic model functionality."""
    print("\nTesting model...")
    
    try:
        from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode
        import numpy as np
        
        # Create a small test configuration
        config = ModelConfig(
            hidden_size=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=32,
            num_layers=1,
            vocab_size=100,
            max_seq_length=32,
            intermediate_size=128  # Smaller for testing
        )
        
        model = Qwen3Model(config)
        print(f"✓ Model initialized with {config.num_layers} layers")
        
        # Test forward pass
        input_ids = np.array([1, 2, 3])
        logits, kv = model.forward(input_ids, InferenceMode.PREFILL)
        
        assert logits.shape[0] == len(input_ids)
        assert logits.shape[1] == config.vocab_size
        print(f"✓ Forward pass successful: {logits.shape}")
        
        # Test generation
        generated = model.generate(input_ids, max_new_tokens=5)
        assert len(generated) > len(input_ids)
        print(f"✓ Generation successful: {len(generated)} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kv_cache():
    """Test KV cache functionality."""
    print("\nTesting KV cache...")
    
    try:
        from python_layer.memory.kv_cache_manager import PagedKVCache
        
        cache = PagedKVCache(
            page_size=8,
            num_pages=16,
            head_dim=32,
            num_kv_heads=2
        )
        
        print(f"✓ KV cache initialized: {cache.num_pages} pages")
        
        # Test allocation
        request_id = 1
        num_pages = 3
        block = cache.allocate_pages(request_id, num_pages)
        
        assert block is not None
        assert len(block.page_ids) == num_pages
        print(f"✓ Allocated {num_pages} pages for request {request_id}")
        
        # Test freeing
        cache.free_pages_for_request(request_id)
        stats = cache.get_cache_memory_usage()
        
        assert stats['free_pages'] >= num_pages
        print(f"✓ Freed pages: {stats['free_pages']} free")
        
        return True
    except Exception as e:
        print(f"✗ KV cache test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MILI Project Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_tokenizer,
        test_model,
        test_kv_cache,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
