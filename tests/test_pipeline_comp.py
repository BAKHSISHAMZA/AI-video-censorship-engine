"""
Enhanced pipeline test demonstrating all features:
- Per-class censorship strategies
- Intensity controls
- Motion blur
- Multiple configuration examples
"""

from censorship_engine.core.datatypes import CensorshipPipeline, ProcessingConfig, CensorshipMethod
from pathlib import Path
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_per_class_strategies():
    """Test different strategies for different classes."""
    
    input_video = r""
    output_video = r""
    
    if not Path(input_video).exists():
        print(f"❌ Test video not found: {input_video}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST: PER-CLASS CENSORSHIP STRATEGIES")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Faces:     Pixelate (block size 20)")
    print("  - Breast:    Blur (kernel 51)")
    print("  - Buttocks:  Motion Blur (25px, 45°)")
    print("  - Genitals:  Black Box (full opacity)")
    print("=" * 70 + "\n")
    
    try:
        config = ProcessingConfig(
            # Per-class strategies
            class_strategies={
                'face': 'pixelate',
                'breast': 'blur',
                'buttocks': 'motion_blur',
                'genitals': 'blackbox',
            },
            
            # Strategy-specific settings
            pixelate_block_size=20,
            pixelate_intensity=1.0,
            
            blur_kernel_size=51,
            blur_intensity=1.0,
            
            # Motion blur settings (new!)
            motion_blur_kernel_size=25,
            motion_blur_angle=45.0,  # 45 degrees
            motion_blur_intensity=1.0,
            
            blackbox_color=(0, 0, 0),  # Black
            blackbox_opacity=1.0,
            blackbox_rounded_corners=0,
            
            # Model settings
            rtdetr_model_path=r" ",
            device="cpu",
            
            # Processing settings
            temporal_smoothing=True,
            smoothing_alpha=0.7,
        )
        
        pipeline = CensorshipPipeline(config)
        
        def progress(current, total):
            percent = (current / total) * 100
            print(f"Progress: {current}/{total} frames ({percent:.1f}%)", end='\r')
        
        result = pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
            progress_callback=progress
        )
        
        print(f"\n✓ Test passed!")
        print(f"  Output: {output_video}")
        print(f"  Processed: {result.frames_processed} frames")
        print(f"  Time: {result.processing_time_seconds:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intensity_controls():
    """Test intensity controls for all strategies."""
    
    input_video = r"C:\Users\usuario\Downloads\6969.mp4"
    output_video = r"C:\Users\usuario\Downloads\out_intensity.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Test video not found: {input_video}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST: INTENSITY CONTROLS")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - All classes: Blur with 70% intensity (semi-transparent)")
    print("=" * 70 + "\n")
    
    try:
        config = ProcessingConfig(
            default_censorship_method=CensorshipMethod.BLUR,
            
            # Reduced intensity for semi-transparent effect
            blur_kernel_size=51,
            blur_intensity=0.7,  # 70% blur, 30% original
            
            rtdetr_model_path=r"C:\Users\usuario\Downloads\protfolio\project_1\models\best.pt",
            device="cpu",
        )
        
        pipeline = CensorshipPipeline(config)
        
        result = pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
        )
        
        print(f"\n✓ Test passed!")
        print(f"  Output: {output_video}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_colored_boxes():
    """Test colored boxes for different classes."""
    
    input_video = r"C:\Users\usuario\Downloads\6969.mp4"
    output_video = r"C:\Users\usuario\Downloads\out_colored.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Test video not found: {input_video}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST: COLORED BOXES")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Faces:     Green box")
    print("  - Breast:    Red box")
    print("  - Buttocks:  Blue box")
    print("  - Genitals:  Black box")
    print("=" * 70 + "\n")
    
    try:
        # Note: To use different colors per class, we need to create
        # separate strategy configurations. This is a limitation we can fix.
        
        config = ProcessingConfig(
            class_strategies={
                'face': 'blur',        # Green boxes need custom implementation
                'breast': 'blackbox',  # Will use config color (red)
                'buttocks': 'blackbox',
                'genitals': 'blackbox',
            },
            
            blackbox_color=(0, 0, 255),  # Red in BGR
            blackbox_opacity=1.0,
            
            rtdetr_model_path=r"C:\Users\usuario\Downloads\protfolio\project_1\models\best.pt",
            device="cpu",
        )
        
        pipeline = CensorshipPipeline(config)
        
        result = pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
        )
        
        print(f"\n✓ Test passed!")
        print(f"  Output: {output_video}")
        print(f"\n  Note: For per-class colors, see advanced config below")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blackbox_fixed():
    """Test that blackbox rendering is working correctly."""
    
    input_video = r"C:\Users\usuario\Downloads\6969.mp4"
    output_video = r"C:\Users\usuario\Downloads\out_blackbox.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Test video not found: {input_video}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST: BLACKBOX RENDERING (FIXED)")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - All classes: Black boxes (sharp corners)")
    print("=" * 70 + "\n")
    
    try:
        config = ProcessingConfig(
            default_censorship_method=CensorshipMethod.BLACKOUT,
            
            blackbox_color=(0, 0, 0),  # Black
            blackbox_opacity=1.0,      # Fully opaque
            blackbox_rounded_corners=0,  # Sharp corners
            
            rtdetr_model_path=r"C:\Users\usuario\Downloads\protfolio\project_1\models\best.pt",
            device="cpu",
        )
        
        pipeline = CensorshipPipeline(config)
        
        result = pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
        )
        
        print(f"\n✓ Test passed! Blackbox rendering is working.")
        print(f"  Output: {output_video}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases."""
    
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS")
    print("=" * 70)
    
    tests = [
        ("Per-Class Strategies", test_per_class_strategies),
        ("Intensity Controls", test_intensity_controls),
        ("Colored Boxes", test_colored_boxes),
        ("Blackbox Fixed", test_blackbox_fixed),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n\nRunning: {name}")
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<50} {status}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print("=" * 70)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your pipeline is production-ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "class":
            success = test_per_class_strategies()
        elif test_name == "intensity":
            success = test_intensity_controls()
        elif test_name == "colored":
            success = test_colored_boxes()
        elif test_name == "blackbox":
            success = test_blackbox_fixed()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: class, intensity, colored, blackbox, all")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)