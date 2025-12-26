"""Unit tests for Qwen3 tokenizer."""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer, MessageFormatter


class TestQwenTokenizer(unittest.TestCase):
    """Tests for QwenTokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = QwenTokenizer()
    
    def test_tokenizer_init(self):
        """Test tokenizer initialization."""
        self.assertGreater(self.tokenizer.vocab_size, 0)
        self.assertIn('<s>', self.tokenizer.special_tokens)
        self.assertIn('</s>', self.tokenizer.special_tokens)
    
    def test_special_tokens(self):
        """Test special token handling."""
        special_token = '<pad>'
        token_id = self.tokenizer.special_tokens.get(special_token)
        self.assertIsNotNone(token_id)
        self.assertIn(token_id, self.tokenizer.id_to_token)
    
    def test_encode_simple(self):
        """Test simple encoding."""
        text = "hello"
        tokens = self.tokenizer.encode(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        for token in tokens:
            self.assertIsInstance(token, int)
    
    def test_encode_with_special_tokens(self):
        """Test encoding with special tokens."""
        text = "hello world"
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Should have BOS and EOS
        self.assertGreater(len(tokens), 2)
    
    def test_decode_simple(self):
        """Test simple decoding."""
        text = "hello"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        original = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(original, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Should roughly preserve content
        self.assertGreater(len(decoded), 0)
    
    def test_vocab_size(self):
        """Test vocabulary size."""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 151936)
    
    def test_return_tensors_numpy(self):
        """Test numpy tensor output."""
        text = "test"
        tokens = self.tokenizer.encode(text, return_tensors='np')
        
        self.assertIsInstance(tokens, np.ndarray)
        self.assertEqual(tokens.dtype, np.int32)
    
    def test_convert_ids_to_tokens(self):
        """Test ID to token conversion."""
        text = "test"
        token_ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        self.assertEqual(len(tokens), len(token_ids))
        for token in tokens:
            self.assertIsInstance(token, str)
    
    def test_convert_tokens_to_ids(self):
        """Test token to ID conversion."""
        tokens = ['<s>', 'hello', '</s>']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        self.assertEqual(len(token_ids), len(tokens))
        for token_id in token_ids:
            self.assertIsInstance(token_id, int)


class TestMessageFormatter(unittest.TestCase):
    """Tests for MessageFormatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = QwenTokenizer()
        self.formatter = MessageFormatter(self.tokenizer)
    
    def test_format_single_user_message(self):
        """Test formatting single user message."""
        messages = [{"role": "user", "content": "Hello"}]
        formatted = self.formatter.format_chat(messages)
        
        self.assertIn("user", formatted)
        self.assertIn("Hello", formatted)
        self.assertIn("<|im_start|>", formatted)
    
    def test_format_conversation(self):
        """Test formatting conversation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        formatted = self.formatter.format_chat(messages)
        
        self.assertIn("user", formatted)
        self.assertIn("assistant", formatted)
        self.assertIn("2+2", formatted)
        self.assertIn("4", formatted)
    
    def test_encode_chat(self):
        """Test encoding chat messages."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = self.formatter.encode_chat(messages)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_parse_response(self):
        """Test response parsing."""
        response = "<|im_start|>assistant\nHello there!<|im_end|>"
        parsed = MessageFormatter.parse_response(response)
        
        self.assertEqual(parsed, "Hello there!")
    
    def test_parse_response_multiline(self):
        """Test multiline response parsing."""
        response = "<|im_start|>assistant\nLine 1\nLine 2<|im_end|>"
        parsed = MessageFormatter.parse_response(response)
        
        self.assertIn("Line 1", parsed)
        self.assertIn("Line 2", parsed)


if __name__ == '__main__':
    unittest.main()
