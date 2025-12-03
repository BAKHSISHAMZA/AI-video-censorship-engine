"""Quick test for both detectors."""

import numpy as np
from censorship_engine.detection import RTDETRDetector, FaceDetector

def test_detectors():
    print("Testing Detectors...")
    print("=" * 60)
    
    # Create dummy frames
    frames = np.random.randint(0, 255, (2, 640, 640, 3), dtype=np.uint8)
    print(f"Created test frames: {frames.shape}")
    
    # Test Face Detector
    print("\n1. Testing FaceDetector...")
    try:
        face_detector = FaceDetector(confidence_threshold=0.5)
        print(f"   ✓ FaceDetector initialized")
        print(f"   Info: {face_detector.get_info()}")
        
        # Warmup
        face_detector.warmup(num_iterations=1)
        print(f"   ✓ Warmup complete")
        
        # Detect
        face_detections = face_detector.detect(frames)
        print(f"   ✓ Detection complete")
        print(f"   Found {sum(len(d) for d in face_detections)} faces")
        
    except Exception as e:
        print(f"   ✗ FaceDetector test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test RT-DETR (if model exists)
    print("\n2. Testing RTDETRDetector...")
    try:
        from pathlib import Path
        model_path = ""
        
        if Path(model_path).exists():
            rtdetr = RTDETRDetector(model_path, device="cpu")
            print(f"   ✓ RTDETRDetector initialized")
            print(f"   Info: {rtdetr.get_info()}")
            
            rtdetr.warmup(num_iterations=1)
            print(f"   ✓ Warmup complete")
            
            nudity_detections = rtdetr.detect(frames)
            print(f"   ✓ Detection complete")
            print(f"   Found {sum(len(d) for d in nudity_detections)} nudity detections")
        else:
            print(f"   ⚠ Model not found: {model_path}")
            print(f"   Skipping RT-DETR test")
            
    except Exception as e:
        print(f"   ✗ RTDETRDetector test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Detector tests complete")

if __name__ == "__main__":
    test_detectors()
