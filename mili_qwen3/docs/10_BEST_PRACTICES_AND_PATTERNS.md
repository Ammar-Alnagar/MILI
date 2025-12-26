# Best Practices and Design Patterns for MILI

## Overview

This guide documents best practices, design patterns, and architectural recommendations for building production-grade LLM inference systems with MILI.

---

## Part 1: Code Organization and Architecture

### 1.1 Module Structure

**Recommended Organization**:

```python
# Good: Clear separation of concerns
mojo_kernels/
├── core/              # Compute kernels
├── memory/            # Memory management
└── utils/             # Type definitions

python_layer/
├── model/             # Model architecture
├── inference/         # Inference logic
├── tokenizer/         # Tokenization
├── server/            # API layer
└── utils/             # Utilities
```

**Anti-pattern: Monolithic files**:

```python
# Bad: All code in one file
utils/
└── everything.py  # 5000+ lines, hard to maintain
```

### 1.2 Dependency Injection

**Pattern: Constructor Injection**

```python
# Good: Dependencies injected
class InferenceEngine:
    def __init__(
        self,
        model: Qwen3Model,
        scheduler: ContinuousBatchScheduler,
        cache: RadixAttentionCache,
        tokenizer: QwenTokenizer
    ):
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.tokenizer = tokenizer

# Usage
engine = InferenceEngine(
    model=model,
    scheduler=scheduler,
    cache=cache,
    tokenizer=tokenizer
)
```

**Anti-pattern: Global state**

```python
# Bad: Hard to test, difficult to reason about
global_model = None
global_scheduler = None

def initialize():
    global global_model, global_scheduler
    global_model = Qwen3Model(...)
    global_scheduler = ContinuousBatchScheduler(...)

def infer(prompt):
    return global_model.generate(prompt)
```

### 1.3 Configuration Management

**Pattern: Dataclass Configuration**

```python
# Good: Type-safe, validated configuration
from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_queue_size: int = 1000
    request_timeout: float = 300.0
    
    def __post_init__(self):
        if self.port < 1024 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")

# Usage
config = ServerConfig(port=8001)
```

**Anti-pattern: String-based configuration**

```python
# Bad: No type safety, error-prone
config = {
    "host": "0.0.0.0",
    "port": "8000",  # String instead of int!
    "workers": "four"  # Won't work
}
```

---

## Part 2: Performance Best Practices

### 2.1 Memory Management

**Pattern: Context Manager for Resources**

```python
# Good: Automatic cleanup
from contextlib import contextmanager

@contextmanager
def gpu_memory_pool(total_size_gb: int):
    """Allocate and manage GPU memory pool."""
    import torch
    pool = torch.empty(total_size_gb * 1024**3 // 4, dtype=torch.float32, device='cuda')
    try:
        yield pool
    finally:
        del pool
        torch.cuda.empty_cache()

# Usage
with gpu_memory_pool(20) as pool:
    # Use pool for allocations
    pass
```

**Anti-pattern: Manual cleanup**

```python
# Bad: Easy to leak memory
pool = torch.empty(...)
# Process ...
# Forgot to cleanup!
```

### 2.2 Batch Processing

**Pattern: Adaptive Batching**

```python
# Good: Adjust batch size based on available memory
class AdaptiveBatcher:
    def __init__(self, target_memory_gb: float = 0.9):
        self.target_memory = target_memory_gb
        self.batch_size = 1
    
    def get_optimal_batch_size(self) -> int:
        """Determine batch size based on available memory."""
        import torch
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        available = total_memory * self.target_memory
        
        # Estimate batch size (rough heuristic)
        batch_size = max(1, int(available / 2.0))  # ~2GB per request
        
        return batch_size

batcher = AdaptiveBatcher()
batch_size = batcher.get_optimal_batch_size()
```

**Anti-pattern: Fixed batch size**

```python
# Bad: Doesn't adapt to hardware
batch_size = 64  # May OOM or underutilize
```

### 2.3 Kernel Optimization

**Pattern: Benchmark-Driven Optimization**

```python
# Good: Measure before and after
import time

def benchmark_kernel(kernel_fn, *args, iterations=100):
    """Benchmark kernel implementation."""
    import torch
    
    # Warmup
    for _ in range(10):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(iterations):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    duration = (time.time() - start) / iterations
    return duration

# Compare implementations
t1 = benchmark_kernel(attention_v1, q, k, v)
t2 = benchmark_kernel(attention_v2, q, k, v)

if t2 < t1:
    print(f"✓ V2 is {t1/t2:.2f}x faster")
else:
    print("✗ V2 is slower, keep V1")
```

**Anti-pattern: Optimization without measurement**

```python
# Bad: Guessing about performance
def optimize():
    # Maybe this is faster? Who knows!
    pass
```

---

## Part 3: Error Handling and Validation

### 3.1 Input Validation

**Pattern: Comprehensive Validation**

```python
# Good: Validate early, fail fast
class RequestValidator:
    def validate_generation_request(self, request: GenerationRequest) -> None:
        """Validate request parameters."""
        errors = []
        
        # Check prompt
        if not request.prompt:
            errors.append("Prompt cannot be empty")
        if len(request.prompt) > 100000:
            errors.append("Prompt too long (max 100K chars)")
        
        # Check tokens
        if request.max_tokens < 1:
            errors.append("max_tokens must be >= 1")
        if request.max_tokens > 32768:
            errors.append("max_tokens exceeds max_position_embeddings")
        
        # Check sampling parameters
        if request.temperature < 0:
            errors.append("temperature must be >= 0")
        if not (0 <= request.top_p <= 1):
            errors.append("top_p must be in [0, 1]")
        
        if errors:
            raise ValueError("; ".join(errors))

# Usage
validator = RequestValidator()
validator.validate_generation_request(request)
```

**Anti-pattern: No validation**

```python
# Bad: Errors occur deep in the system
def process_request(request):
    # Crash somewhere in the model
    return model.generate(request.prompt)
```

### 3.2 Graceful Degradation

**Pattern: Fallback Strategies**

```python
# Good: Handle failures gracefully
def generate_with_fallback(
    prompt: str,
    strategies: List[callable] = None
) -> str:
    """Try multiple strategies, fallback on failure."""
    if strategies is None:
        strategies = [
            lambda: model.generate(prompt),  # Try with full model
            lambda: small_model.generate(prompt),  # Fallback to smaller
            lambda: cached_response(prompt),  # Use cached
        ]
    
    for strategy in strategies:
        try:
            return strategy()
        except Exception as e:
            print(f"Strategy failed: {e}, trying next...")
            continue
    
    raise RuntimeError("All strategies exhausted")

# Usage
result = generate_with_fallback(prompt)
```

**Anti-pattern: Crash on first error**

```python
# Bad: No fallback
return model.generate(prompt)  # Dies if model fails
```

---

## Part 4: Testing Strategies

### 4.1 Unit Testing Pattern

**Pattern: Test Fixtures and Parametrization**

```python
import pytest
from python_layer.tokenizer.qwen_tokenizer import QwenTokenizer

@pytest.fixture
def tokenizer():
    """Fixture for tokenizer."""
    return QwenTokenizer()

@pytest.mark.parametrize("text,expected_len", [
    ("Hello", 1),
    ("Hello world", 2),
    ("", 0),
])
def test_tokenize(tokenizer, text, expected_len):
    """Test tokenization with multiple inputs."""
    tokens = tokenizer.encode(text)
    assert len(tokens) >= expected_len

def test_roundtrip(tokenizer):
    """Test encode/decode roundtrip."""
    original = "The quick brown fox"
    tokens = tokenizer.encode(original)
    decoded = tokenizer.decode(tokens)
    
    # Allow some whitespace differences
    assert original.lower() in decoded.lower()
```

### 4.2 Integration Testing Pattern

**Pattern: End-to-End Testing**

```python
import pytest
import torch

class TestInferenceEndToEnd:
    """End-to-end inference tests."""
    
    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests."""
        from python_layer.model.config import Qwen3Config
        from python_layer.model.weight_loader import WeightLoader
        from python_layer.model.qwen3_model import Qwen3Model
        
        config = Qwen3Config(hidden_size=2048, num_hidden_layers=8)
        loader = WeightLoader("./models/test", config)
        return Qwen3Model(config, loader)
    
    def test_generation_produces_valid_tokens(self, model):
        """Test that generation produces valid tokens."""
        output = model.generate("Hello", max_tokens=10)
        
        # Check output is string
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_generation_respects_max_tokens(self, model):
        """Test that generation respects max_tokens."""
        output = model.generate("Test", max_tokens=5)
        
        # Rough check (token count ~= word count / 1.3)
        token_estimate = len(output.split()) * 1.3
        assert token_estimate <= 10  # 5 tokens + some margin
    
    def test_deterministic_with_temperature_zero(self, model):
        """Test deterministic behavior with temperature=0."""
        # Temperature 0 should always produce same output
        prompt = "Deterministic test"
        
        # Would need model to support this
        # output1 = model.generate(prompt, temperature=0)
        # output2 = model.generate(prompt, temperature=0)
        # assert output1 == output2
```

### 4.3 Performance Testing Pattern

**Pattern: Benchmark Suite**

```python
import time
import statistics

class BenchmarkSuite:
    """Benchmark suite for performance tracking."""
    
    def __init__(self, iterations: int = 10):
        self.iterations = iterations
        self.results = {}
    
    def benchmark_attention(self, seq_len: int, batch_size: int):
        """Benchmark attention kernel."""
        import torch
        
        q = torch.randn(batch_size, 32, seq_len, 64, device='cuda')
        k = torch.randn(batch_size, 32, seq_len, 64, device='cuda')
        v = torch.randn(batch_size, 32, seq_len, 64, device='cuda')
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            # Call attention kernel
            # output = attention(q, k, v)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        }
    
    def compare_implementations(self, v1, v2, *args):
        """Compare two implementations."""
        t1 = self.benchmark_attention(*args)
        t2 = self.benchmark_attention(*args)
        
        speedup = t1["mean"] / t2["mean"]
        print(f"V2 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than V1")
        
        return speedup
```

---

## Part 5: Logging and Monitoring

### 5.1 Structured Logging

**Pattern: Contextual Logging**

```python
import logging
from contextlib import contextmanager
from functools import wraps

# Setup structured logger
logger = logging.getLogger(__name__)

@contextmanager
def log_context(operation: str, **context_vars):
    """Context manager for structured logging."""
    logger.info(f"Starting: {operation}", extra=context_vars)
    try:
        yield
    except Exception as e:
        logger.error(f"Failed: {operation}", exc_info=True, extra=context_vars)
        raise
    else:
        logger.info(f"Completed: {operation}", extra=context_vars)

def log_performance(func):
    """Decorator to log performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.info(
            f"{func.__name__} took {duration:.3f}s",
            extra={
                "function": func.__name__,
                "duration_ms": duration * 1000
            }
        )
        return result
    return wrapper

# Usage
with log_context("inference", request_id="req-123", batch_size=64):
    result = model.generate(prompt)

@log_performance
def process_request(request):
    return model.generate(request.prompt)
```

### 5.2 Metrics Collection

**Pattern: Metrics Registry**

```python
from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class MetricsCollector:
    """Collect and report system metrics."""
    
    metrics: Dict[str, list] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict:
        """Get metrics summary."""
        import statistics
        
        summary = {}
        for name, values in self.metrics.items():
            if not values:
                continue
            
            summary[name] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }
        
        return summary
    
    def print_summary(self):
        """Print metrics summary."""
        summary = self.get_summary()
        print("\n=== Metrics Summary ===")
        for metric, stats in summary.items():
            print(f"{metric}:")
            for key, val in stats.items():
                print(f"  {key}: {val:.2f}")

# Usage
metrics = MetricsCollector()

for request in requests:
    start = time.time()
    result = model.generate(request.prompt)
    duration = time.time() - start
    metrics.record_metric("generation_time_ms", duration * 1000)

metrics.print_summary()
```

---

## Part 6: API Design Best Practices

### 6.1 Request/Response Models

**Pattern: Pydantic Models**

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class GenerationRequest(BaseModel):
    """Generation request with validation."""
    
    prompt: str = Field(..., min_length=1, max_length=100000)
    max_tokens: int = Field(128, ge=1, le=32768)
    temperature: float = Field(1.0, ge=0, le=2.0)
    top_p: float = Field(0.95, ge=0, le=1.0)
    top_k: int = Field(50, ge=0)
    stream: bool = False
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()

class GenerationResponse(BaseModel):
    """Generation response."""
    
    request_id: str
    generated_text: str
    tokens: List[int]
    total_tokens: int
    latency_ms: float

# Usage
request = GenerationRequest(
    prompt="Hello",
    max_tokens=50
)
# Validation happens automatically
```

### 6.2 Error Response Format

**Pattern: Consistent Error Responses**

```python
from pydantic import BaseModel
from enum import Enum

class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_LOADED = "model_not_loaded"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"

class ErrorResponse(BaseModel):
    """Consistent error response format."""
    
    error_code: ErrorCode
    message: str
    details: Optional[dict] = None
    request_id: Optional[str] = None

# Usage in FastAPI
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code=ErrorCode.INVALID_REQUEST,
            message=str(exc),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )
```

---

## Part 7: Documentation Best Practices

### 7.1 Docstring Convention

**Pattern: Comprehensive Docstrings**

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50
) -> str:
    """
    Generate text from a prompt.
    
    This method generates tokens autoregressively using the Qwen3 model.
    It supports various sampling strategies including top-k and nucleus
    (top-p) sampling for controlling output diversity.
    
    Args:
        prompt: Input text prompt to continue. Must be non-empty.
        max_tokens: Maximum number of tokens to generate.
            Range: [1, max_position_embeddings]. Default: 128.
        temperature: Sampling temperature for diversity control.
            Range: [0, 2]. Lower values = more deterministic,
            higher values = more random. Default: 1.0.
        top_p: Nucleus sampling parameter. Keep tokens with cumulative
            probability <= top_p. Range: [0, 1]. Default: 0.95.
        top_k: Top-k sampling parameter. Keep only top-k most likely
            tokens. Default: 50.
    
    Returns:
        Generated text string.
    
    Raises:
        ValueError: If prompt is empty or parameters are invalid.
        RuntimeError: If model not loaded or CUDA OOM.
        TimeoutError: If generation exceeds timeout.
    
    Example:
        >>> model = Qwen3Model(config, weight_loader)
        >>> output = model.generate(
        ...     "What is AI?",
        ...     max_tokens=100,
        ...     temperature=0.7
        ... )
        >>> print(output)
    
    Note:
        This is an autoregressive generation process and may take
        several seconds for large max_tokens values.
    """
```

---

## Part 8: Security Best Practices

### 8.1 Input Sanitization

**Pattern: Input Validation and Sanitization**

```python
import re
from typing import Optional

class InputSanitizer:
    """Sanitize user inputs."""
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Remove potentially harmful content from prompt."""
        # Remove null bytes
        prompt = prompt.replace('\x00', '')
        
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Limit length
        max_length = 100000
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
        
        return prompt.strip()
    
    @staticmethod
    def validate_request_id(request_id: str) -> bool:
        """Validate request ID format."""
        # UUID format check
        return bool(re.match(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            request_id
        ))

# Usage
sanitized = InputSanitizer.sanitize_prompt(user_input)
```

### 8.2 Rate Limiting

**Pattern: Rate Limiter**

```python
from collections import defaultdict
import time

class RateLimiter:
    """Rate limiting for API requests."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = defaultdict(list)  # IP -> [timestamp]
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if client is allowed to make request."""
        now = time.time()
        one_minute_ago = now - 60
        
        # Remove old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if t > one_minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.rpm:
            return False
        
        # Record request
        self.requests[client_ip].append(now)
        return True

# Usage in FastAPI
@app.post("/generate")
async def generate(request: GenerationRequest, request_obj: Request):
    client_ip = request_obj.client.host
    if not limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # ... process request
```

---

## Part 9: Deployment Best Practices

### 9.1 Configuration Management

**Pattern: Environment-Based Configuration**

```python
import os
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Config:
    """Base configuration."""
    
    ENV = Environment(os.getenv("ENV", "development"))
    DEBUG = ENV == Environment.DEVELOPMENT
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    MAX_BATCH_SIZE = 4
    WORKER_THREADS = 1

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    MAX_BATCH_SIZE = 256
    WORKER_THREADS = 8
    ENABLE_MONITORING = True

# Usage
if Config.ENV == Environment.PRODUCTION:
    config = ProductionConfig()
else:
    config = DevelopmentConfig()
```

### 9.2 Health Checks and Readiness

**Pattern: Comprehensive Health Checks**

```python
from enum import Enum
import torch

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    """Check system health."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def check_health(self) -> dict:
        """Perform comprehensive health check."""
        status = HealthStatus.HEALTHY
        checks = {}
        
        # Check GPU
        try:
            torch.cuda.current_device()
            checks["gpu"] = "ok"
        except Exception as e:
            checks["gpu"] = f"error: {e}"
            status = HealthStatus.DEGRADED
        
        # Check model
        if self.model is not None:
            checks["model"] = "loaded"
        else:
            checks["model"] = "not_loaded"
            status = HealthStatus.UNHEALTHY
        
        # Check tokenizer
        if self.tokenizer is not None:
            checks["tokenizer"] = "ready"
        else:
            checks["tokenizer"] = "not_ready"
            status = HealthStatus.UNHEALTHY
        
        # Check memory
        try:
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            checks["gpu_memory_gb"] = f"{gpu_mem:.2f}"
        except:
            checks["gpu_memory_gb"] = "unknown"
        
        return {
            "status": status,
            "checks": checks
        }

# Usage in FastAPI
@app.get("/health")
def health_check():
    result = checker.check_health()
    
    if result["status"] == HealthStatus.UNHEALTHY:
        return JSONResponse(result, status_code=503)
    
    return result
```

---

## Summary Table

| Topic | Best Practice | Anti-Pattern |
|-------|---------------|--------------|
| Architecture | Separation of concerns | Monolithic code |
| Dependencies | Injection | Global state |
| Configuration | Type-safe dataclass | String-based dicts |
| Memory | Context managers | Manual cleanup |
| Testing | Comprehensive suites | No tests |
| Logging | Structured logging | Print statements |
| Errors | Graceful degradation | Crash on error |
| API | Validated models | Raw strings |
| Documentation | Comprehensive docstrings | No docs |
| Security | Input validation | Trust user input |
| Deployment | Environment-based config | Hardcoded values |

---

## Checklist for Production Readiness

- [ ] All critical functions have tests
- [ ] Logging is comprehensive and structured
- [ ] Error handling is graceful with fallbacks
- [ ] Configuration is environment-specific
- [ ] Health checks are implemented
- [ ] Rate limiting is enabled
- [ ] Input validation is comprehensive
- [ ] Documentation is complete
- [ ] Performance is benchmarked
- [ ] Security measures are in place
- [ ] Monitoring is set up
- [ ] Graceful shutdown is implemented

---

## References

- [Clean Code by Robert Martin](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)
- [Design Patterns by Gang of Four](https://en.wikipedia.org/wiki/Design_Patterns)
- [Python Best Practices](https://pep8.org/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/concepts/)
