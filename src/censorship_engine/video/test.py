"""Manual test for video I/O."""
import sys
import sys
from pathlib import Path
import numpy as np

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent / "src"))
from censorship_engine.video.loader import VideoLoader
from censorship_engine.video.writer import VideoWriter


def test_with_real_video(video_path: str):
    """Test loading and writing a real video."""
    
    print(f"\n{'='*60}")
    print(f"Testing with video: {video_path}")
    print(f"{'='*60}\n")
    
    # Test VideoLoader
    print("1. Testing VideoLoader...")
    try:
        with VideoLoader(video_path, batch_size=8) as loader:
            print(f"   ✓ Video loaded successfully")
            print(f"   - Resolution: {loader.width}x{loader.height}")
            print(f"   - FPS: {loader.fps:.2f}")
            print(f"   - Total frames: {loader.frame_count}")
            print(f"   - Batch size: {loader.batch_size}")
            
            print("\n2. Iterating through batches...")
            batch_count = 0
            total_frames = 0
            
            for frame_ids, frames_batch in loader:
                batch_count += 1
                total_frames += len(frame_ids)
                print(f"   Batch {batch_count}: {len(frame_ids)} frames, shape: {frames_batch.shape}")
                
                # Only process first 3 batches for quick test
                if batch_count >= 3:
                    print(f"   ... (stopping after 3 batches for quick test)")
                    break
            
            print(f"\n   ✓ Processed {batch_count} batches, {total_frames} frames total")
            
            # Get stats
            stats = loader.get_stats()
            print(f"\n3. Loader Statistics:")
            print(f"   - Frames read: {stats['frames_read']}")
            print(f"   - Frames skipped (stride): {stats['frames_skipped_stride']}")
            print(f"   - Frames skipped (quality): {stats['frames_skipped_quality']}")
            
    except Exception as e:
        print(f"   ✗ VideoLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test VideoWriter
    print(f"\n4. Testing VideoWriter...")
    output_path = "test_output.mp4"
    
    try:
        with VideoLoader(video_path, batch_size=8) as loader:
            with VideoWriter(
                output_path,
                fps=loader.fps,
                width=loader.width,
                height=loader.height
            ) as writer:
                print(f"   ✓ VideoWriter initialized")
                
                # Write first 3 batches
                batch_count = 0
                for frame_ids, frames_batch in loader:
                    writer.write_batch(frames_batch)
                    batch_count += 1
                    print(f"   Written batch {batch_count}: {len(frame_ids)} frames")
                    
                    #if batch_count >= 3:
                    #    break
                
                stats = writer.get_stats()
                print(f"\n   ✓ Video written successfully")
                print(f"   - Output: {output_path}")
                print(f"   - Frames written: {stats['frames_written']}")
        
        # Verify output exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / 1024 / 1024
            print(f"   - File size: {file_size:.2f} MB")
            print(f"\n   ✓ Output video file created successfully")
        else:
            print(f"\n   ✗ Output file not found!")
            return False
            
    except Exception as e:
        print(f"   ✗ VideoWriter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*60}")
    print(f"✓ ALL TESTS PASSED")
    print(f"{'='*60}\n")
    
    return True


def create_sample_video(output_path: str = "sample_test.mp4"):
    """Create a simple test video if you don't have one."""
    
    print(f"Creating sample video: {output_path}")
    
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    # Create 90 frames (3 seconds at 30 FPS)
    for i in range(90):
        # Create colorful frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 3) % 256  # Blue channel
        frame[:, :, 1] = (i * 5) % 256  # Green channel
        frame[:, :, 2] = (i * 7) % 256  # Red channel
        
        # Add some text
        cv2.putText(
            frame, 
            f"Frame {i}", 
            (50, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2, 
            (255, 255, 255), 
            3
        )
        
        writer.write(frame)
    
    writer.release()
    print(f"✓ Sample video created: {output_path}")
    return output_path


if __name__ == "__main__":
    # Option 1: Test with your own video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
    else:
        # Option 2: Create a sample video
        print("No video provided, creating sample video...")
        video_path = create_sample_video()
    
    # Run test
    success = test_with_real_video(video_path)
    
    sys.exit(0 if success else 1)