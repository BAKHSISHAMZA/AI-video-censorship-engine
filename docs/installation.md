# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with 6+ GB VRAM (recommended)
- **Storage**: 10 GB free space

### Recommended Requirements
- **GPU**: NVIDIA Tesla T4, RTX 3060, or better
- **RAM**: 16 GB
- **Storage**: 20 GB SSD

## Installation Steps

### 1. Install Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS (using Homebrew)
brew install python@3.10

# Windows
# Download from python.org
```

### 2. Install CUDA (GPU support)
```bash
# Ubuntu 20.04/22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda-11-8

# Verify installation
nvidia-smi
```

### 3. Clone Repository
```bash
git clone https://github.com/yourusername/ai-video-censorship-engine.git
cd ai-video-censorship-engine
```

### 4. Create Virtual Environment
```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 5. Install Dependencies
```bash
# Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 6. Download Models
```bash
# Run download script
bash models/download_models.sh

# Or manually download
# Place rtdetr_nudity.pt in models/ directory
```

### 7. Verify Installation
```bash
# Run tests
pytest tests/

# Test on sample video
python scripts/censor_video.py examples/sample.mp4 output.mp4
```

## Troubleshooting

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print GPU name
```

If False:
- Check NVIDIA driver: `nvidia-smi`
- Reinstall CUDA-enabled PyTorch
- Check CUDA version compatibility

### Out of Memory Errors

Reduce batch size in config:
```yaml
performance:
  batch_size: 8  # Reduce from 16
```

### FFmpeg Not Found
```bash
# Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### Model Download Fails

Download manually from [releases page] and place in `models/` directory.

## Docker Installation (Alternative)
```bash
# Build image
docker build -t censorship-engine .

# Run container
docker run --gpus all -v $(pwd)/videos:/videos censorship-engine \
  censor-video /videos/input.mp4 /videos/output.mp4
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Configuration Guide](configuration.md)
- [API Reference](api_reference.md)