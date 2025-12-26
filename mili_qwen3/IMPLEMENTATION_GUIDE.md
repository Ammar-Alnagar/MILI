# MILI Implementation Guide

## Overview

MILI (Mojo Inference Language Engine) is a production-ready inference server for Qwen3 language models using Mojo kernels. It provides a simple, efficient API for text generation with GPU acceleration.

## Architecture

### 1. Core Components

#### Server Layer (`server.py`)
- **FastAPI Application**: RESTful API for text generation
- **Request Handling**: Input validation and response formatting
- **Health Monitoring**: System status and model loading checks

#### Model Integration
- **HuggingFace Transformers**: Direct integration with official Qwen3 models
- **AutoTokenizer**: Automatic tokenization with proper encoding/decoding
- **AutoModelForCausalLM**: Optimized inference with GPU acceleration

#### Configuration (`config/`)
- **Model Config**: Model parameters and settings (legacy)
- **Inference Config**: Generation parameters and optimization settings

### 2. Key Features

#### HuggingFace Integration
- Direct use of official Qwen3 models from HuggingFace Hub
- Automatic model downloading and caching
- Support for different model sizes and variants

#### GPU Acceleration
- Automatic CUDA detection and device placement
- Optimized inference with transformers backend
- CPU fallback for systems without GPU

#### RESTful API
- FastAPI-based server with OpenAPI documentation
- Request validation with Pydantic models
- Health check and monitoring endpoints

#### Flexible Generation
- Configurable sampling parameters (temperature, top_p, top_k)
- Adjustable maximum token limits
- Proper token counting and response formatting

## Implementation Details

### Server Architecture

The server follows a simple, production-ready pattern:

```python
# server.py structure
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="MILI Qwen3 Server")

# Global model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

@app.post("/generate")
async def generate(request: GenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
```

### Model Loading Process

```
User Request -> FastAPI Server -> Model Loading (startup)
                                       ↓
HuggingFace Hub -> Download Model -> Cache Locally -> GPU/CPU
                                       ↓
Ready for Inference -> Handle Requests -> Generate Text
```

### Text Generation Flow

```
Input Prompt -> Tokenization -> Model Forward Pass -> Sampling -> Detokenization
      ↓              ↓                ↓              ↓            ↓
   "Hello"    [151643, 1234]    logits tensor   top_p sampling   "Hello world"
```

## Usage

### Starting the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

The server will start on `http://localhost:9999` and automatically download the Qwen3-0.6B model.

### API Usage

#### Text Generation

```python
import requests

response = requests.post("http://localhost:9999/generate", json={
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
})

result = response.json()
print(result["generated_text"])
```

#### Health Check

```python
response = requests.get("http://localhost:9999/health")
status = response.json()
print(f"Model loaded: {status['model_loaded']}")
```

### Python Integration

For programmatic usage, you can integrate the model directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Generate text
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Configuration

The server uses these default generation parameters:

```python
DEFAULT_CONFIG = {
    "max_tokens": 50,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50
}
```

You can override these in each request.

## Performance Considerations

### Hardware Optimization
1. **GPU Usage**: Automatic CUDA detection and utilization
2. **Memory Management**: Efficient tensor operations with PyTorch
3. **Batch Processing**: Single request processing (extendable to batching)

### Generation Parameters
1. **Temperature**: Controls randomness (0.1-2.0)
2. **Top-p**: Nucleus sampling for quality (0.1-0.99)
3. **Top-k**: Limits token selection (1-100)
4. **Max Tokens**: Response length control (1-4096)

### Expected Performance (Qwen3-0.6B)
- **Generation Speed**: 20-50 tokens/second (GPU)
- **Memory Usage**: 2-4GB GPU VRAM, 4-8GB system RAM
- **Latency**: 2-10 seconds for 100-200 token responses
- **Concurrent Users**: 1-5 simultaneous requests (depending on hardware)

## Testing

### Basic Functionality Test
```bash
# Test the server health
curl http://localhost:9999/health

# Test text generation
curl -X POST http://localhost:9999/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 20}'
```

### Integration Tests
```bash
python -m pytest tests/integration/test_inference.py -v
```

### Unit Tests
```bash
python -m pytest tests/unit/test_tokenizer.py -v
```

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

### Production Server
```bash
# Use a production ASGI server
pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 9999

CMD ["python", "server.py"]
```

```bash
# Build and run
docker build -t mili-qwen3 .
docker run -p 9999:9999 mili-qwen3
```

### Integration Tests
```bash
python -m pytest tests/integration/test_inference.py
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## Deployment

### Requirements
- CUDA 11.8+
- Mojo 0.1.0+
- Python 3.8+
- numpy, torch (optional)

### Installation
```bash
pip install -r requirements.txt
cd mojo_kernels && bash build.sh
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
RUN apt-get install -y mojo python3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check internet connection
ping huggingface.co

# Clear cache if corrupted
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B

# Try different model
# Change "Qwen/Qwen3-0.6B" to "Qwen/Qwen3-1.7B" in server.py
```

#### CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Force CPU usage (edit server.py)
device = "cpu"  # Instead of torch.cuda.is_available()

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Reduce generation length
# Set max_tokens to lower value in requests
```

#### Port Conflicts
```bash
# Find process using port
lsof -i :9999

# Kill conflicting process
kill -9 <PID>

# Or change port in server.py
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Performance Tuning

#### Optimize for Speed
- Use GPU if available
- Lower max_tokens for faster responses
- Adjust temperature (lower = faster, less creative)

#### Optimize for Quality
- Increase temperature for more creative responses
- Use top_p < 0.9 for focused generation
- Experiment with different prompts

## Future Enhancements

1. **Batch Processing**: Support multiple concurrent requests
2. **Model Variants**: Support for different Qwen model sizes
3. **Streaming Responses**: Real-time token streaming
4. **Quantization**: 4-bit/8-bit model quantization
5. **Multi-GPU**: Distributed inference across GPUs
6. **Caching**: Request/result caching for repeated queries

## References

- [Qwen3 Official Documentation](https://huggingface.co/Qwen)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
