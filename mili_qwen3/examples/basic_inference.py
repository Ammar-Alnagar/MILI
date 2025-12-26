"""
Basic inference example for Qwen3 model.
Demonstrates how to use the MILI framework for LLM inference.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_layer import (
    Qwen3Model,
    ModelConfig,
    QwenTokenizer,
    MessageFormatter,
    PagedKVCache,
    InferenceEngine,
    InferenceRequest,
)


def example_basic_generation():
    """Example: Basic token generation."""
    print("=" * 60)
    print("Example 1: Basic Token Generation")
    print("=" * 60)

    # Initialize model
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        intermediate_size=1024,
        num_layers=2,
        max_seq_length=512,
    )

    model = Qwen3Model(config)
    print(f"Model parameters: ~{model.config.num_parameters:,}")

    # Generate tokens
    prompt = np.array([1, 2, 3, 4, 5])
    print(f"Prompt: {prompt}")

    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    print(f"Generated: {generated}")
    print(f"New tokens: {len(generated) - len(prompt)}")


def example_tokenization():
    """Example: Text tokenization."""
    print("\n" + "=" * 60)
    print("Example 2: Text Tokenization")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = QwenTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Encode text
    text = "Hello, how are you doing today?"
    tokens = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")

    # Decode tokens
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Decoded: {decoded}")


def example_chat_format():
    """Example: Chat message formatting."""
    print("\n" + "=" * 60)
    print("Example 3: Chat Message Formatting")
    print("=" * 60)

    tokenizer = QwenTokenizer()
    formatter = MessageFormatter(tokenizer)

    # Create a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you explain deep learning?"},
    ]

    # Format and encode
    formatted = formatter.format_chat(messages)
    tokens = formatter.encode_chat(messages)

    print("Formatted chat:")
    print(formatted[:200] + "...")
    print(f"\nTokenized length: {len(tokens)} tokens")


def example_kv_cache():
    """Example: KV cache management."""
    print("\n" + "=" * 60)
    print("Example 4: KV Cache Management")
    print("=" * 60)

    # Initialize cache
    cache = PagedKVCache(
        page_size=16,
        num_pages=64,
        head_dim=128,
        num_kv_heads=8,
        enable_prefix_sharing=True,
    )

    print(f"Cache pages: {cache.num_pages}")
    print(f"Page size: {cache.page_size} tokens")

    # Allocate for a request
    request_id = 1
    num_pages = 4
    block = cache.allocate_pages(request_id, num_pages)

    print(f"Allocated {num_pages} pages for request {request_id}")

    # Write KV data
    seq_len = 32
    k_data = np.random.randn(seq_len, 8, 128).astype(np.float32)
    v_data = np.random.randn(seq_len, 8, 128).astype(np.float32)

    cache.write_kv(request_id, k_data, v_data)
    print(f"Wrote {seq_len} tokens to cache")

    # Read back
    k_read, v_read = cache.read_kv(request_id, num_tokens=16)
    print(f"Read {k_read.shape[0]} tokens from cache")

    # Check memory usage
    stats = cache.get_cache_memory_usage()
    print(f"Cache usage:")
    print(f"  Used pages: {stats['used_pages']}")
    print(f"  Free pages: {stats['free_pages']}")
    print(f"  Total memory: {stats['total_used_bytes'] / 1024:.1f} KB")

    # Free cache
    cache.free_pages(request_id)
    print(f"Freed cache for request {request_id}")


def example_inference_engine():
    """Example: Inference engine with continuous batching."""
    print("\n" + "=" * 60)
    print("Example 5: Inference Engine")
    print("=" * 60)

    # Setup
    config = ModelConfig(
        hidden_size=128,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        vocab_size=100,
        max_seq_length=256,
    )

    model = Qwen3Model(config)
    cache = PagedKVCache(
        page_size=16,
        num_pages=32,
        head_dim=config.head_dim,
        num_kv_heads=config.num_kv_heads,
    )

    engine = InferenceEngine(model, cache, batch_size=4)
    print(f"Engine batch size: {engine.batch_size}")

    # Add requests
    for i in range(3):
        request = InferenceRequest(
            request_id=i, input_ids=[1, 2, 3, 4], max_new_tokens=5
        )
        engine.add_request(request)

    print(f"Added 3 requests to engine")

    # Execute steps
    for step in range(10):
        completed = engine.step()

        if completed:
            for req_id, output in completed:
                print(f"Request {req_id} completed: {output.token_ids}")

        if not engine.active_requests and not engine.pending_requests:
            break

    # Get stats
    stats = engine.get_stats()
    print(f"\nEngine stats:")
    print(f"  Completed: {stats['completed_requests']}")
    print(f"  Cache used: {stats['cache_memory']['used_pages']} pages")


def example_batch_processing():
    """Example: Batch processing with continuous batching."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)

    config = ModelConfig(
        hidden_size=64,
        num_heads=2,
        num_kv_heads=1,
        num_layers=1,
        vocab_size=50,
        max_seq_length=128,
    )

    model = Qwen3Model(config)
    tokenizer = QwenTokenizer()

    # Multiple prompts
    prompts = [
        "The future of AI is",
        "Machine learning can help",
        "Neural networks are",
    ]

    print(f"Processing {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i + 1}. {prompt}")

    # Generate for each prompt
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = np.array(tokens[:10], dtype=np.int32)  # Limit to 10 tokens

        generated = model.generate(input_ids, max_new_tokens=5)
        print(f"Generated {len(generated) - len(input_ids)} tokens")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MILI Framework - Qwen3 Inference Examples")
    print("=" * 60)

    try:
        example_tokenization()
        example_basic_generation()
        example_chat_format()
        example_kv_cache()
        example_inference_engine()
        example_batch_processing()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
