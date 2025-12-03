# test_pipeline_fixed.py

from censorship_engine import CensorshipPipeline, ProcessingConfig, CensorshipMethod
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Use a video with ACTUAL content to detect
input_video = r""  # Your raw uncensored video

config = ProcessingConfig(
    nudity_confidence_threshold=0.45,
    face_confidence_threshold=0.50,
    censorship_method=CensorshipMethod.BLUR,
    rtdetr_model_path=r"",
    device="cpu",
    batch_size=2,
    preserve_audio=True,  
)
output_path =  r""
pipeline = CensorshipPipeline(config)
result = pipeline.process_video(input_video, output_path)

print(f"\nDetections: {result.total_detections}")
print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")