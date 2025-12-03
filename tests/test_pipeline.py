"""End-to-end pipeline test with progress."""

from censorship_engine import CensorshipPipeline, ProcessingConfig, CensorshipMethod
from pathlib import Path
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_full_pipeline():
    """Test complete pipeline on a video."""
    
    input_video = " "
    output_video = " "
    
    if not Path(input_video).exists():
        print(f"❌ Test video not found: {input_video}")
        return
    
    print("\n" + "=" * 70)
    print("STARTING FULL PIPELINE TEST")
    print("=" * 70)
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Device: CPU (processing will be slower)")
    print("=" * 70 + "\n")
    
    try:
        # Create config
        config = ProcessingConfig(
            class_strategies={
                'face': 'motion_blur',        # Green boxes need custom implementation
                'breast': 'blackbox',  # Will use config color (red)
                'buttocks': 'blackbox',
                'genitals': 'blackbox',
            },
            # Motion blur settings (new!)
            # Tracking parameters
            max_track_age = 5,
            min_track_hits = 1,
            iou_threshold = 0.5,
            motion_blur_kernel_size=25,
            motion_blur_angle=45.0,  # 45 degrees
            motion_blur_intensity=1.0,
            blackbox_color=(0, 0, 0),  # black in BGR
            blackbox_opacity=1.0,
            
            rtdetr_model_path=" ",
            device="cpu",
        )
        
        # Create pipeline
        print("Initializing pipeline components...")
        pipeline = CensorshipPipeline(config)
        print("✓ Pipeline initialized\n")
        
        # Progress callback
        def progress(current, total):
            percent = (current / total) * 100
            print(f"Progress: {current}/{total} frames ({percent:.1f}%)", end='\r')
        
        # Process video
        print("Starting video processing...\n")
        result = pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
            progress_callback=progress
        )
        
        # Print results
        print("\n\n" + "=" * 70)
        print("✓ PIPELINE TEST SUCCESSFUL!")
        print("=" * 70)
        print(f"\nInput:  {result.input_path}")
        print(f"Output: {result.output_path}")
        
        print(f"\n📊 Processing Statistics:")
        print(f"  Total frames:      {result.total_frames}")
        print(f"  Frames processed:  {result.frames_processed}")
        print(f"  Frames skipped:    {result.frames_skipped}")
        print(f"  Processing time:   {result.processing_time_seconds:.1f}s")
        print(f"  Average FPS:       {result.fps:.1f}")
        
        print(f"\n🎯 Detection Statistics:")
        print(f"  Total detections:  {result.total_detections}")
        print(f"  Nudity detections: {result.nudity_detections}")
        print(f"  Face detections:   {result.face_detections}")
        print(f"  Total tracks:      {result.total_tracks}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"  Avg detection time: {result.avg_detection_time_ms:.1f}ms")
        print(f"  Avg tracking time:  {result.avg_tracking_time_ms:.1f}ms")
        print(f"  Avg rendering time: {result.avg_rendering_time_ms:.1f}ms")
        
        if result.metadata_path:
            print(f"\n📄 Metadata: {result.metadata_path}")
        
        if result.errors:
            print(f"\n⚠️  Warnings/Errors:")
            for error in result.errors:
                print(f"    - {error}")
        
        print("\n" + "=" * 70)
        print(f"✓ Output video saved: {output_video}")
        print("=" * 70 + "\n")
        
        # Verify output file exists
        if Path(output_video).exists():
            file_size = Path(output_video).stat().st_size / (1024 * 1024)
            print(f"✓ Output file verified: {file_size:.2f} MB")
        else:
            print("❌ Warning: Output file not found")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n\n❌ PIPELINE TEST FAILED")
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Your pipeline is working.")
        print("Next steps:")
        print("  1. Try with different videos")
        print("  2. Test different censorship methods")
        print("  3. Test with GPU (when available)")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")