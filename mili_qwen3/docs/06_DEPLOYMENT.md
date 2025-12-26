# MILI Deployment Guide

# MILI Qwen3 Deployment Guide

This guide covers deploying the MILI Qwen3 inference server in various environments.

## Local Development

### Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Server**
```bash
python server.py
```

3. **Test the API**
```bash
curl -X POST "http://localhost:9999/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 20}'
```

### Requirements Check

```bash
# Check Python version
python --version  # Should be 3.8+

# Check available memory (GPU recommended)
nvidia-smi  # For NVIDIA GPUs

# Check disk space (models need ~5GB)
df -h
```

## Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 9999

CMD ["python", "server.py"]
```

**Build and Run:**
```bash
# Build image
docker build -t mili-qwen3 .

# Run container
docker run -p 9999:9999 mili-qwen3

# Run with GPU support
docker run --gpus all -p 9999:9999 mili-qwen3
```

### Systemd Service

**Create service file** (`/etc/systemd/system/mili-qwen3.service`):
```ini
[Unit]
Description=MILI Qwen3 Inference Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mili-qwen3
ExecStart=/home/ubuntu/venv/bin/python server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Service Management:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start mili-qwen3

# Enable auto-start
sudo systemctl enable mili-qwen3

# Check status
sudo systemctl status mili-qwen3

# View logs
journalctl -u mili-qwen3 -f
```

### Nginx Reverse Proxy

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:9999;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Cloud Deployment

#### AWS EC2 with GPU

1. **Launch EC2 Instance**
   - Instance type: `g4dn.xlarge` (T4 GPU) or `p3.2xlarge` (V100 GPU)
   - AMI: Ubuntu 22.04 LTS
   - Storage: 50GB+ GP3

2. **Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install CUDA (for GPU instances)
# Follow NVIDIA instructions for your GPU type
```

3. **Deploy Application**
```bash
# Clone repository
git clone <your-repo>
cd mili-qwen3

# Install dependencies
pip install -r requirements.txt

# Run with systemd (see above)
```

#### Google Cloud Platform

1. **Create VM Instance**
   - Machine type: `n1-standard-8` or GPU instance
   - GPU: NVIDIA Tesla T4 (optional)
   - OS: Ubuntu 22.04 LTS

2. **Setup Firewall**
```bash
gcloud compute firewall-rules create mili-qwen3 \
  --allow tcp:9999 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow MILI Qwen3 traffic"
```

#### DigitalOcean Droplet

1. **Create Droplet**
   - Image: Ubuntu 22.04 LTS
   - Plan: Basic with 4GB RAM minimum
   - Region: Closest to your users

2. **Setup Domain** (optional)
```bash
# Add A record pointing to droplet IP
# Configure nginx reverse proxy as above
```

## Monitoring

### Health Checks

```bash
# Check server health
curl http://localhost:9999/health

# Monitor with cron
echo "*/5 * * * * curl -f http://localhost:9999/health > /dev/null || systemctl restart mili-qwen3" | crontab -
```

### Performance Monitoring

```python
# monitor.py
import time
import requests
import psutil

def monitor_server():
    # Check response time
    start = time.time()
    response = requests.post("http://localhost:9999/generate",
                           json={"prompt": "Hello", "max_tokens": 10})
    latency = time.time() - start

    # Check system resources
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent

    print(f"Latency: {latency:.2f}s, CPU: {cpu}%, Memory: {memory}%")

    return response.status_code == 200

if __name__ == "__main__":
    monitor_server()
```

### Logging

The server logs to stdout. For production:

```bash
# Log to file
python server.py > server.log 2>&1 &

# Rotate logs
logrotate -f /etc/logrotate.d/mili-qwen3
```

## Scaling

### Vertical Scaling

- **CPU**: More cores help with concurrent requests
- **RAM**: 16GB+ recommended for model loading
- **GPU**: T4, A100, or H100 for best performance
- **Storage**: SSD recommended for model caching

### Horizontal Scaling

For high-traffic applications:

1. **Load Balancer** (nginx/haproxy)
```nginx
upstream mili_backends {
    server 127.0.0.1:9999;
    server 127.0.0.1:10000;
    server 127.0.0.1:10001;
}
```

2. **Multiple Instances**
```bash
# Run multiple servers on different ports
python server.py --port 9999 &
python server.py --port 10000 &
python server.py --port 10001 &
```

3. **Container Orchestration**
```yaml
# docker-compose.yml
version: '3.8'
services:
  mili-qwen3:
    image: mili-qwen3
    deploy:
      replicas: 3
    ports:
      - "9999-10001:9999"
```

## Security

### Basic Security

1. **Firewall Configuration**
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable
```

2. **SSL/TLS with Let's Encrypt**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com
```

3. **API Rate Limiting** (future enhancement)
```python
# Add to server.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

## Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Check disk space
df -h

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
python server.py
```

#### CUDA Out of Memory
```bash
# Use CPU fallback
# Edit server.py: device = "cpu"

# Or reduce model size
# Change to Qwen/Qwen3-0.6B if using larger model
```

#### Port Already in Use
```bash
# Find process
lsof -i :9999

# Kill process
kill -9 <PID>

# Or change port
python server.py --port 8000
```

#### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Ensure model is on GPU: verify `device = "cuda"`
- Reduce max_tokens for faster responses
- Use lower temperature for simpler generation
