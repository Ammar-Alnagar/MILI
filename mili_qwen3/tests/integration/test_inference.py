"""Integration tests for Qwen3 inference pipeline."""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_layer.model.qwen_model import Qwen3Model, ModelConfig, InferenceMode
from python_layer.memory.kv_cache_manager import PagedKVCache, RadixAttentionCache
from python_layer.inference.inference_engine import InferenceEngine, InferenceRequest


class TestQwen3Model(unittest.TestCase):
    """Tests for Qwen3 model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            num_layers=2,
            vocab_size=1000,
            max_seq_length=512
        )
        self.model = Qwen3Model(self.config)
    
    def test_model_init(self):
        """Test model initialization."""
        self.assertEqual(self.model.config.hidden_size, 256)
        self.assertEqual(self.model.config.num_heads, 4)
        self.assertIsNotNone(self.model.embeddings)
    
    def test_forward_pass_prefill(self):
        """Test forward pass in prefill mode."""
        batch_size = 1
        seq_len = 10
        input_ids = np.random.randint(0, 100, size=seq_len)
        
        logits, kv = self.model.forward(input_ids, InferenceMode.PREFILL)
        
        self.assertEqual(logits.shape[0], seq_len)
        self.assertEqual(logits.shape[1], self.config.vocab_size)
        self.assertIsNotNone(kv)
    
    def test_forward_pass_decode(self):
        """Test forward pass in decode mode."""
        input_id = np.array([42])
        
        logits, kv = self.model.forward(input_id, InferenceMode.DECODE)
        
        self.assertEqual(logits.shape[1], self.config.vocab_size)
        self.assertIsNotNone(kv)
    
    def test_kv_cache_update(self):
        """Test KV cache updates."""
        seq_len = 5
        input_ids = np.arange(seq_len)
        
        # Prefill
        logits1, kv1 = self.model.forward(input_ids, InferenceMode.PREFILL)
        
        # Decode with cache
        new_token = np.array([10])
        logits2, kv2 = self.model.forward(new_token, InferenceMode.DECODE, kv1)
        
        self.assertIsNotNone(kv2)
    
    def test_generate_short_sequence(self):
        """Test short sequence generation."""
        prompt = np.array([1, 5, 10])
        max_tokens = 5
        
        generated = self.model.generate(prompt, max_new_tokens=max_tokens)
        
        self.assertGreaterEqual(len(generated), len(prompt))
        self.assertLessEqual(len(generated), len(prompt) + max_tokens)


class TestPagedKVCache(unittest.TestCase):
    """Tests for paged KV cache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = PagedKVCache(
            page_size=16,
            num_pages=32,
            head_dim=64,
            num_kv_heads=8
        )
    
    def test_cache_init(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.page_size, 16)
        self.assertEqual(self.cache.num_pages, 32)
        self.assertEqual(len(self.cache.free_pages), 32)
    
    def test_allocate_pages(self):
        """Test page allocation."""
        request_id = 1
        num_pages = 4
        
        block = self.cache.allocate_pages(request_id, num_pages)
        
        self.assertIsNotNone(block)
        self.assertEqual(len(block.page_ids), num_pages)
        self.assertEqual(len(self.cache.free_pages), 32 - num_pages)
    
    def test_free_pages(self):
        """Test page deallocation."""
        request_id = 1
        num_pages = 4
        
        self.cache.allocate_pages(request_id, num_pages)
        initial_free = len(self.cache.free_pages)
        
        self.cache.free_pages_for_request(request_id)
        
        self.assertEqual(len(self.cache.free_pages), initial_free + num_pages)
    
    def test_write_read_kv(self):
        """Test writing and reading KV data."""
        request_id = 1
        num_pages = 2
        
        self.cache.allocate_pages(request_id, num_pages)
        
        # Create KV data
        k_data = np.random.randn(20, 8, 64).astype(np.float32)
        v_data = np.random.randn(20, 8, 64).astype(np.float32)
        
        self.cache.write_kv(request_id, k_data, v_data)
        
        # Read back
        k_read, v_read = self.cache.read_kv(request_id, num_tokens=20)
        
        self.assertEqual(k_read.shape, k_data.shape)
        self.assertEqual(v_read.shape, v_data.shape)
    
    def test_cache_memory_usage(self):
        """Test cache memory usage reporting."""
        request_id = 1
        self.cache.allocate_pages(request_id, 4)
        
        stats = self.cache.get_cache_memory_usage()
        
        self.assertIn('used_pages', stats)
        self.assertIn('free_pages', stats)
        self.assertEqual(stats['used_pages'], 4)
        self.assertEqual(stats['free_pages'], 28)


class TestRadixAttentionCache(unittest.TestCase):
    """Tests for RadixAttention cache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.paged_cache = PagedKVCache(
            page_size=16,
            num_pages=64,
            head_dim=64,
            num_kv_heads=8
        )
        self.radix_cache = RadixAttentionCache(self.paged_cache)
    
    def test_radix_init(self):
        """Test radix cache initialization."""
        self.assertIsNotNone(self.radix_cache.paged_cache)
        self.assertTrue(self.radix_cache.enable_prefix_sharing)
    
    def test_allocate_with_prefix_sharing(self):
        """Test allocation with prefix sharing."""
        request_id = 1
        prompt_tokens = [1, 2, 3, 4, 5]
        
        block = self.radix_cache.allocate_with_prefix_sharing(
            request_id,
            prompt_tokens
        )
        
        self.assertIsNotNone(block)
        self.assertEqual(block.request_id, request_id)
    
    def test_prefix_matching(self):
        """Test prefix matching."""
        request_id1 = 1
        request_id2 = 2
        
        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 6, 7]
        
        existing = {request_id1: tokens1}
        
        block = self.radix_cache.allocate_with_prefix_sharing(
            request_id2,
            tokens2,
            existing
        )
        
        self.assertIsNotNone(block)


class TestInferenceEngine(unittest.TestCase):
    """Tests for inference engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            hidden_size=128,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            num_layers=1,
            vocab_size=100,
            max_seq_length=256
        )
        self.model = Qwen3Model(self.config)
        self.kv_cache = PagedKVCache(
            page_size=16,
            num_pages=32,
            head_dim=self.config.head_dim,
            num_kv_heads=self.config.num_kv_heads
        )
        self.engine = InferenceEngine(self.model, self.kv_cache)
    
    def test_engine_init(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.kv_cache)
        self.assertEqual(len(self.engine.active_requests), 0)
    
    def test_add_request(self):
        """Test adding request."""
        request = InferenceRequest(
            request_id=1,
            input_ids=[1, 2, 3],
            max_new_tokens=10
        )
        
        self.engine.add_request(request)
        
        self.assertEqual(len(self.engine.pending_requests), 1)
    
    def test_get_stats(self):
        """Test statistics reporting."""
        stats = self.engine.get_stats()
        
        self.assertIn('active_requests', stats)
        self.assertIn('pending_requests', stats)
        self.assertIn('cache_memory', stats)


class TestEndToEndGeneration(unittest.TestCase):
    """End-to-end generation tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            hidden_size=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=32,
            num_layers=1,
            vocab_size=50,
            max_seq_length=128
        )
        self.model = Qwen3Model(self.config)
    
    def test_simple_generation(self):
        """Test simple token generation."""
        prompt = np.array([1, 2, 3])
        max_tokens = 5
        
        generated = self.model.generate(prompt, max_new_tokens=max_tokens)
        
        self.assertGreater(len(generated), len(prompt))
        self.assertLessEqual(len(generated) - len(prompt), max_tokens)
    
    def test_generation_deterministic_seed(self):
        """Test deterministic generation with seed."""
        np.random.seed(42)
        prompt = np.array([1, 2, 3])
        
        generated1 = self.model.generate(prompt, max_new_tokens=5)
        
        np.random.seed(42)
        generated2 = self.model.generate(prompt, max_new_tokens=5)
        
        np.testing.assert_array_equal(generated1, generated2)


if __name__ == '__main__':
    unittest.main()
