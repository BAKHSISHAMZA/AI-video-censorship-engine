# Quick Start Tutorial

This guide will walk you through your first video censorship in 5 minutes.

## Prerequisites

- Installed the engine (see [Installation Guide](installation.md))
- Have a test video ready

## Step 1: Basic Censorship

Process a video with default settings:
```bash
python scripts/censor_video.py input.mp4 output.mp4
```

This will:
- Detect nudity (confidence > 0.45)
- Detect faces (confidence > 0.50)
- Apply Gaussian blur censorship
- Save censored video to output.mp4

## Step 2: Adjust Sensitivity

Make detection more sensitive (catch more content):
```bash
python scripts/censor_video.py input.mp4 output.mp4 --threshold 0.3
```

Lower threshold = more detections (including false positives)

## Step 3: Use Preset Configuration

Use the strict preset (maximum censorship):
```bash
python scripts/censor_video.py input.mp4 output.mp4 --config config/presets/strict.yaml
```

Available presets:
- `strict.yaml` - Maximum recall, more false positives
- `balanced.yaml` - Good tradeoff (default)
- `precision.yaml` - Minimize false positives

## Step 4: Python API

Use the engine programmatically:
```python
from censorship_engine.pipeline import CensorshipPipeline
from censorship_engine.core.datatypes import ProcessingConfig

# Create config
config = ProcessingConfig(
    nudity_confidence_threshold=0.45,
    batch_size=16
)

# Create pipeline
pipeline = CensorshipPipeline(config)

# Process video
result = pipeline.process_video("input.mp4", "output.mp4")

# Print results
print(f"Processed {result.frames_processed} frames")
print(f"Found {result.total_detections} detections")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

## Step 5: Customize Censorship

Change censorship method:
```python
from censorship_engine.core.datatypes import CensorshipMethod

config = ProcessingConfig(
    censorship_method=CensorshipMethod.PIXELATE,
    pixelate_block_size=32  # Larger blocks
)

pipeline = CensorshipPipeline(config)
result = pipeline.process_video("input.mp4", "output.mp4")
```

Available methods:
- `BLUR` - Gaussian blur (default)
- `PIXELATE` - Pixelation/mosaic
- `BLACKOUT` - Solid black boxes
- `EMOJI` - Emoji overlays (fun mode)

## Step 6: Batch Processing

Process multiple videos:
```python
from pathlib import Path

video_dir = Path("videos/")
output_dir = Path("censored/")
output_dir.mkdir(exist_ok=True)

for video_path in video_dir.glob("*.mp4"):
    output_path = output_dir / f"censored_{video_path.name}"
    result = pipeline.process_video(str(video_path), str(output_path))
    print(f"Processed: {video_path.name}")
```

## Step 7: Export Metadata

Get detailed detection metadata:
```python
config = ProcessingConfig(
    export_metadata=True,
    metadata_format="json"
)

result = pipeline.process_video("input.mp4", "output.mp4")
print(f"Metadata saved to: {result.metadata_path}")
```

Metadata includes:
- All detection coordinates
- Confidence scores
- Track IDs
- Timestamps
- Performance metrics

## Common Use Cases

### Case 1: Content Moderation
```python
# Strict detection, blackout censorship
config = ProcessingConfig(
    nudity_confidence_threshold=0.30,
    censorship_method=CensorshipMethod.BLACKOUT
)
```

### Case 2: Privacy Protection
```python
# Only blur faces, keep nudity detection disabled
config = ProcessingConfig(
    nudity_confidence_threshold=1.0,  # Effectively disabled
    face_confidence_threshold=0.40,
    censorship_method=CensorshipMethod.BLUR,
    blur_kernel_size=75  # Heavy blur
)
```

### Case 3: Fast Preview
```python
# Process every 5th frame for quick preview
config = ProcessingConfig(
    skip_frames=4,  # Process every 5th frame
    batch_size=32   # Larger batches
)
```

## Next Steps

- [Configuration Reference](configuration.md) - All config options
- [API Documentation](api_reference.md) - Full API details
- [Architecture Overview](architecture.md) - How it works
- [Performance Tuning](benchmarks.md) - Optimize for your hardware
