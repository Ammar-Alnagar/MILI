"""
Qwen3 Tokenizer Implementation.
Provides tokenization and detokenization for Qwen3 model.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Token:
    """Represents a single token."""
    id: int
    text: str
    type: str  # "normal", "special", "bpe"


class QwenTokenizer:
    """Qwen3 tokenizer based on BPE."""
    
    # Special tokens
    SPECIAL_TOKENS = {
        '<|im_start|>': 151857,
        '<|im_end|>': 151858,
        '<|im_sep|>': 151859,
        '<pad>': 151643,
        '<unk>': 151643,
        '</s>': 151645,
        '<s>': 151644,
    }
    
    def __init__(
        self,
        vocab_size: int = 151936,
        special_tokens: Optional[Dict[str, int]] = None,
        merge_rules: Optional[Dict[Tuple[str, str], str]] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            special_tokens: Mapping of special tokens
            merge_rules: BPE merge rules
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or self.SPECIAL_TOKENS.copy()
        self.merge_rules = merge_rules or {}
        
        # Build vocab
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary mappings."""
        # Add special tokens
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        # Add character tokens (basic vocab)
        next_id = 256
        for i in range(256):
            if i not in self.id_to_token:
                char = chr(i)
                if char.isprintable() or char in '\n\t':
                    self.token_to_id[char] = i
                    self.id_to_token[i] = char
                    next_id = max(next_id, i + 1)
        
        # Fill remaining vocab with placeholder BPE tokens
        while next_id < self.vocab_size:
            self.id_to_token[next_id] = f"<bpe_{next_id}>"
            self.token_to_id[f"<bpe_{next_id}>"] = next_id
            next_id += 1
    
    @classmethod
    def from_pretrained(cls, vocab_path: Path) -> "QwenTokenizer":
        """Load tokenizer from pretrained vocabulary file."""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            return cls(
                vocab_size=vocab_data.get('vocab_size', 151936),
                special_tokens=vocab_data.get('special_tokens', cls.SPECIAL_TOKENS),
                merge_rules=vocab_data.get('merge_rules', {})
            )
        except FileNotFoundError:
            # Default tokenizer if file not found
            return cls()
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Return format ("pt", "np", None)
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        # Add BOS if requested
        if add_special_tokens:
            tokens.append(self.special_tokens.get('<s>', 1))
        
        # Split by special tokens first
        remaining_text = text
        
        for special_token in self.special_tokens.keys():
            if special_token in remaining_text:
                parts = remaining_text.split(special_token)
                tokens.extend(self._encode_text(parts[0]))
                
                if special_token in self.token_to_id:
                    tokens.append(self.token_to_id[special_token])
                
                remaining_text = ''.join(parts[1:])
        
        # Encode remaining text
        tokens.extend(self._encode_text(remaining_text))
        
        # Add EOS if requested
        if add_special_tokens:
            tokens.append(self.special_tokens.get('</s>', 2))
        
        if return_tensors == 'np':
            import numpy as np
            return np.array(tokens, dtype=np.int32)
        elif return_tensors == 'pt':
            try:
                import torch
                return torch.tensor(tokens, dtype=torch.int32)
            except ImportError:
                return tokens
        
        return tokens
    
    def _encode_text(self, text: str) -> List[int]:
        """Encode a piece of text to token IDs using BPE."""
        if not text:
            return []
        
        # Start with character tokens
        tokens = [ord(c) for c in text]
        
        # Apply merge rules (simplified BPE)
        for _ in range(len(self.merge_rules)):
            if len(tokens) <= 1:
                break
            
            # Find most frequent pair
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
            
            most_frequent = max(pairs, key=pairs.get)
            
            # Merge most frequent pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_frequent:
                    # Find or create merged token
                    merged_text = f"{self.id_to_token.get(tokens[i], chr(tokens[i]))}{self.id_to_token.get(tokens[i+1], chr(tokens[i+1]))}"
                    if merged_text not in self.token_to_id:
                        new_id = len(self.token_to_id)
                        self.token_to_id[merged_text] = new_id
                        self.id_to_token[new_id] = merged_text
                    
                    new_tokens.append(self.token_to_id[merged_text])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            clean_up_tokenization_spaces: Clean up extra spaces
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                
                tokens.append(token)
            else:
                tokens.append(f"<unk_{token_id}>")
        
        text = ''.join(tokens)
        
        # Clean up tokenization spaces
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
        
        return text
    
    @staticmethod
    def _clean_up_tokenization(text: str) -> str:
        """Clean up tokenization artifacts."""
        # Remove extra spaces
        text = text.replace('  ', ' ')
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' )', ')')
        text = text.replace('( ', '(')
        
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get special tokens mapping."""
        return self.special_tokens.copy()
    
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to tokens."""
        return [self.id_to_token.get(token_id, f"<unk_{token_id}>") for token_id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to token IDs."""
        return [self.token_to_id.get(token, self.token_to_id.get('<unk>', 0)) for token in tokens]


class MessageFormatter:
    """Format messages for Qwen3 chat model."""
    
    def __init__(self, tokenizer: QwenTokenizer):
        """Initialize formatter with tokenizer."""
        self.tokenizer = tokenizer
    
    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for model input.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts
            
        Returns:
            Formatted chat string
        """
        formatted = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                formatted.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == 'assistant':
                formatted.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == 'system':
                formatted.append(f"<|im_start|>system\n{content}<|im_end|>")
        
        # Add assistant prompt for next response
        formatted.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted)
    
    def encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode formatted chat messages."""
        formatted = self.format_chat(messages)
        return self.tokenizer.encode(formatted)
    
    @staticmethod
    def parse_response(text: str) -> str:
        """Parse assistant response from formatted text."""
        # Remove formatting markers
        if '<|im_start|>assistant' in text:
            text = text.split('<|im_start|>assistant\n')[-1]
        
        if '<|im_end|>' in text:
            text = text.split('<|im_end|>')[0]
        
        return text.strip()
