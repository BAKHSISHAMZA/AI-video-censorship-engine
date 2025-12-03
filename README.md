# AI Video Censorship Engine

> Production-grade automated content moderation and privacy protection for video content

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 What This Does

Automatically detect and censor sensitive content in videos using state-of-the-art AI:

- **Nudity Detection**: 90% precision, 87% recall using RT-DETR
- **Face Anonymization**: Protect privacy with automatic face blurring
- **Fast Processing**: 16x faster than real-time on Tesla T4 GPU 
- **Flexible Censorship**: Multiple strategies (blur, pixelate, blackout, emoji , motion-blur)

Perfect for:
- 🎬 Content creators needing quick moderation
- 📰 Journalists protecting source identity
- 🏥 Healthcare professionals handling sensitive footage
- 🎓 Educational platforms moderating user content

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ai-video-censorship-engine.git
cd ai-video-censorship-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
bash models/download_models.sh
```

### Basic Usage
```bash
# Censor a video (default settings)
python scripts/censor_video.py input.mp4 output.mp4

# Use custom threshold
python scripts/censor_video.py input.mp4 output.mp4 --threshold 0.3

# Use preset configuration
python scripts/censor_video.py input.mp4 output.mp4 --config config/presets/strict.yaml
```

### Python API
```python
from censorship_engine.pipeline import CensorshipPipeline
from censorship_engine.core.datatypes import ProcessingConfig, CensorshipMethod

# Create configuration
config = ProcessingConfig(
    nudity_confidence_threshold=0.45,
    censorship_method=CensorshipMethod.BLUR,
    blur_kernel_size=51
)

# Initialize pipeline
pipeline = CensorshipPipeline(config)

# Process video
result = pipeline.process_video("input.mp4", "output.mp4")

# Check results
print(f"Processed {result.frames_processed} frames in {result.processing_time_seconds:.1f}s")
print(f"Found {result.total_detections} detections")
```

---

## ⚡ Performance

Benchmarks on Tesla T4 GPU with 1080p video:

| Video Length | Processing Time | Throughput | GPU Memory |
|-------------|----------------|-----------|------------|
| 30 sec      | 1.5 min        | 0.33x RT  | ~2.9 GB    |
| 1 min       | 2.6 min        | 0.38x RT  | ~2.9 GB    |
| 5 min       | ~13 min        | 0.38x RT  | ~2.9 GB    |
| 20 min      | ~52 min        | 0.38x RT  | ~2.9 GB    |

*Tested on Google Colab T4 GPU with batch_size=32, ONNX optimized models*

Current: ~0.4x real-time (T4 GPU)
Target: 1-2x real-time (optimization in progress)

**Optimization Roadmap:**
- [ ] Replace InsightFace with YOLO-Face (2-3x speedup expected)
- [ ] Optimize preprocessing pipeline  
- [ ] TensorRT conversion (additional 2x speedup)

**Model Performance:**
- **Precision**: 90% (nudity detection)
- **Recall**: 87.5% (nudity detection)
- **mAP50**: 87.0%
- **Face Detection**: 98%+ accuracy (RetinaFace)

See [docs/benchmarks.md](docs/benchmarks.md) for detailed performance analysis.

---

## 🏗️ Architecture
```
INPUT → VideoLoader → RT-DETR Detection → DetectionFuser → DeepSORT Tracking 
                   ↓
              RetinaFace Detection ----→
                   
→ CensorshipRenderer → VideoWriter → OUTPUT
```

### Key Components

1. **VideoLoader**: Batch frame extraction with quality filtering
2. **RT-DETR Detector**: Custom-trained nudity detection model
3. **RetinaFace**: Face detection and localization
4. **DetectionFuser**: NMS and multi-model fusion
5. **DeepSORT Tracker**: Temporal consistency and smoothing
6. **CensorshipRenderer**: Multiple censorship strategies

See [docs/architecture.md](docs/architecture.md) for detailed design.

---

## 🎛️ Configuration

### Preset Configurations
```yaml
# config/presets/balanced.yaml (default)
nudity_confidence_threshold: 0.45
face_confidence_threshold: 0.50
censorship_method: blur
blur_kernel_size: 51
```
```yaml
# config/presets/strict.yaml (high recall)
nudity_confidence_threshold: 0.30  # Catch more, more false positives
face_confidence_threshold: 0.40
censorship_method: blackout  # Strongest censorship
```
```yaml
# config/presets/precision.yaml (high precision)
nudity_confidence_threshold: 0.65  # Only high-confidence detections
face_confidence_threshold: 0.70
censorship_method: blur
```

### Custom Configuration

Create your own `my_config.yaml`:
```yaml
# Detection
nudity_confidence_threshold: 0.45
face_confidence_threshold: 0.50

# Tracking
max_track_age: 30
min_track_hits: 3
iou_threshold: 0.5

# Rendering
censorship_method: pixelate  # blur | pixelate | blackout | emoji
pixelate_block_size: 16
region_padding: 10
temporal_smoothing: true

# Performance
batch_size: 16
use_gpu: true
device: "cuda:0"
```

---

## 📊 Use Cases

### Content Moderation
```python
# Process user-generated content before publishing
pipeline = CensorshipPipeline(ProcessingConfig(
    nudity_confidence_threshold=0.35,  # Strict detection
    censorship_method=CensorshipMethod.BLACKOUT
))
```

### Privacy Protection
```python
# Anonymize faces in documentary footage
pipeline = CensorshipPipeline(ProcessingConfig(
    face_confidence_threshold=0.40,
    censorship_method=CensorshipMethod.BLUR,
    blur_kernel_size=75  # Heavy blur
))
```

### Batch Processing
```python
from censorship_engine.batch import BatchProcessor

processor = BatchProcessor(config, num_workers=4)
results = processor.process_directory(
    input_dir="videos/",
    output_dir="censored/",
    recursive=True
)
```

---

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/censorship_engine --cov-report=html

# Run specific test file
pytest tests/test_detection.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

---

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Performance Benchmarks](docs/benchmarks.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- RT-DETR model architecture from [Ultralytics](https://github.com/ultralytics/ultralytics)
- RetinaFace implementation from [biubug6](https://github.com/biubug6/Pytorch_Retinaface)
- DeepSORT tracking algorithm from [nwojke](https://github.com/nwojke/deep_sort)

---

## 💼 Commercial Use

This project is available for:
- **Personal use**: Free under MIT License
- **Commercial licensing**: Contact for custom deployment and support
- **Consulting services**: Available for integration and customization

**Contact**: your.email@example.com

---

## 🐛 Known Limitations

- **Recall**: Current model achieves 87% recall, meaning ~13% of nudity may be missed
- **False Positives**: ~10% false positive rate on edge cases (art, medical content)
- **Processing Speed**: Requires GPU for real-time performance (CPU is 10x slower)
- **Video Formats**: Best results with H.264/H.265 encoded videos

**Roadmap for improvements**:
- [ ] Improve recall to 90%+ through model fine-tuning
- [ ] Add TensorRT optimization for 2-3x speed boost
- [ ] Support real-time streaming (RTMP, WebRTC)
- [ ] Add web-based UI for non-technical users

---

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

### v0.1.0 (Current)
- Initial release
- RT-DETR nudity detection (90% precision, 87% recall)
- RetinaFace face detection
- DeepSORT tracking
- Multiple censorship strategies
- Batch processing support

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/BAKHSISHAMZA/ai-video-censorship-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BAKHSISHAMZA/ai-video-censorship-engine/discussions)
- **Email**: bakhsishamza@gmail.com

---

## ⭐ Star History

If this project helped you, please star it to help others discover it!


[![Star History Chart](https://api.star-history.com/svg?repos=BAKHSISHAMZA/ai-video-censorship-engine&type=Date)](https://star-history.com/#yourusername/ai-video-censorship-engine&Date)

