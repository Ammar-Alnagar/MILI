# MILI API Reference Documentation

# MILI Qwen3 API Reference

Complete API reference for the MILI Qwen3 inference server.

## REST API Endpoints

### Base URL
```
http://localhost:9999
```

### Health Check

**GET** `/health`

Returns server status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_ready": true,
  "model_type": "Qwen3-0.6B",
  "vocab_size": 151936,
  "max_seq_length": 40960
}
```

### Text Generation

**POST** `/generate`

Generate text from a prompt using the Qwen3 model.

**Request Body:**
```json
{
  "prompt": "What is machine learning?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

**Parameters:**
- `prompt` (string, required): Input text prompt
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 50, max: 4096)
- `temperature` (float, optional): Sampling temperature (default: 1.0, range: 0.1-2.0)
- `top_p` (float, optional): Nucleus sampling (default: 0.95, range: 0.1-0.99)
- `top_k` (integer, optional): Top-k sampling (default: 50, range: 1-100)

**Response:**
```json
{
  "generated_text": "Machine learning is a subset of artificial intelligence...",
  "tokens": [151643, 1234, 5678, ...],
  "total_tokens": 150,
  "prompt_tokens": 5
}
```

### Root Endpoint

**GET** `/`

Returns server information and available endpoints.

**Response:**
```json
{
  "message": "MILI Qwen3 Inference Server",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/health",
  "generate": "/generate (POST)"
}
```

## Request/Response Models

### GenerationRequest

```python
class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
```

### GenerationResponse

```python
class GenerationResponse(BaseModel):
    """Text generation response."""
    generated_text: str
    tokens: List[int]
    total_tokens: int
    prompt_tokens: int
```

## Usage Examples

### Python Client

```python
import requests

# Basic generation
response = requests.post("http://localhost:9999/generate", json={
    "prompt": "Explain quantum computing:",
    "max_tokens": 200,
    "temperature": 0.7
})
result = response.json()
print(result["generated_text"])

# Advanced parameters
response = requests.post("http://localhost:9999/generate", json={
    "prompt": "Write a poem about AI:",
    "max_tokens": 100,
    "temperature": 1.2,  # More creative
    "top_p": 0.9,        # Nucleus sampling
    "top_k": 40          # Top-k limit
})
```

### cURL Examples

```bash
# Basic request
curl -X POST "http://localhost:9999/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is deep learning?", "max_tokens": 50}'

# Advanced parameters
curl -X POST "http://localhost:9999/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain neural networks",
    "max_tokens": 150,
    "temperature": 0.5,
    "top_p": 0.8
  }'

# Health check
curl http://localhost:9999/health
```

### JavaScript Client

```javascript
// Using fetch
async function generateText(prompt) {
  const response = await fetch('http://localhost:9999/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 100,
      temperature: 0.7
    })
  });

  const result = await response.json();
  return result.generated_text;
}

// Usage
generateText("Hello, how are you?").then(text => {
  console.log(text);
});
```

## Error Responses

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **422**: Validation Error (malformed request)
- **500**: Internal Server Error (generation failed)
- **503**: Service Unavailable (model not loaded)

**Error Response Format:**
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

Current implementation has no rate limiting (single-threaded). For production use, consider:

- Request queuing
- Concurrent request limits
- API key authentication
- Usage monitoring

## Model Information

**Current Model**: Qwen/Qwen3-0.6B

- **Parameters**: 0.6 billion
- **Vocabulary**: 151,936 tokens
- **Context Length**: 40,960 tokens
- **Architecture**: Decoder-only transformer with GQA
- **Training**: Pre-trained by Alibaba Cloud

