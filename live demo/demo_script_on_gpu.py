""" this was created to test on google colab T4 free GPU """
# Add src to Python path
import sys
import os
sys.path.insert(0, '/content/ai-video-censorship-engine/src')

# Import the pipeline
from censorship_engine import AsyncOptimizedPipeline, ProcessingConfig, CensorshipMethod
import torch

# Verify CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Create config
config = ProcessingConfig(
            class_strategies={
                'face': 'motion_blur',        # Green boxes need custom implementation
                'breast': 'blur',  # Will use config color (red)
                'buttocks': 'blur',
                'genitals': 'blackbox',
            },
            # Motion blur settings (new!)
            motion_blur_kernel_size=25,
            motion_blur_angle=55.0,  # 45 degrees
            motion_blur_intensity=1.0,
            blur_kernel_size=60,
            blur_intensity=1.0,
            blackbox_color=(0, 0, 0),  # black in BGR
            blackbox_opacity=1.0,
            # Strategy-specific settings
            pixelate_block_size=20,
            pixelate_intensity=1.0,
            # Tracking parameters
            max_track_age = 5,
            min_track_hits = 1,
            iou_threshold = 0.5,

            rtdetr_model_path= ''# in .onnx format ,
            device="cuda",
            batch_size = 28,
            nudity_confidence_threshold = 0.35 ,
            face_confidence_threshold = 0.5,
            region_padding = 40

        )


# Initialize pipeline
print("\nInitializing pipeline...")
pipeline = AsyncOptimizedPipeline(config)

# Set paths
input_video = ''  
output_video = ''

# Process video
print(f"\nProcessing: {input_video}")
print(f"Output will be saved to: {output_video}")
print("\nStarting processing... (this may take a while)")

try:
    result = pipeline.process_video(
        input_path=input_video,
        output_path=output_video
    )
    print(f"\n✓ SUCCESS! Output saved to: {result}")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()