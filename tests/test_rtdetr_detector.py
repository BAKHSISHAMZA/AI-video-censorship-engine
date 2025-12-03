"""Tests for RT-DETR detector."""
import sys
import os
import pytest
import numpy as np
from pathlib import Path

from censorship_engine.detection.rtdetr_detector import (
    RTDETRDetector,
    RTDETRDetectorFactory,
    create_detector
)
from censorship_engine.core.exceptions import ModelLoadError, DetectionError


class TestRTDETRDetector:
    """Test RT-DETR detector functionality."""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock model file."""
        model_path = tmp_path / "test_model.pt"
        model_path.touch()
        return str(model_path)
    
    def test_initialization_with_invalid_path(self):
        """Test that initialization fails with non-existent model."""
        with pytest.raises(ModelLoadError):
            RTDETRDetector("nonexistent_model.pt")
    
    def test_invalid_confidence_threshold(self, mock_model_path):
        """Test that invalid confidence threshold raises error."""
        with pytest.raises(ValueError):
            RTDETRDetector(mock_model_path, confidence_threshold=1.5)
        
        with pytest.raises(ValueError):
            RTDETRDetector(mock_model_path, confidence_threshold=-0.1)
    
    def test_invalid_iou_threshold(self, mock_model_path):
        """Test that invalid IoU threshold raises error."""
        with pytest.raises(ValueError):
            RTDETRDetector(mock_model_path, iou_threshold=2.0)
    
    def test_detect_with_invalid_input_shape(self, mock_model_path):
        """Test that detect fails with wrong input shape."""
        detector = RTDETRDetector(mock_model_path)
        
        # 3D array instead of 4D
        frames = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect(frames)
        
        # Wrong number of channels
        frames = np.zeros((2, 640, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect(frames)
    
    def test_set_confidence_threshold(self, mock_model_path):
        """Test updating confidence threshold."""
        detector = RTDETRDetector(mock_model_path, confidence_threshold=0.5)
        
        detector.set_confidence_threshold(0.3)
        assert detector.confidence_threshold == 0.3
        
        with pytest.raises(ValueError):
            detector.set_confidence_threshold(1.5)
    
    def test_get_info(self, mock_model_path):
        """Test getting detector info."""
        detector = RTDETRDetector(
            mock_model_path,
            confidence_threshold=0.45,
            device="cpu"
        )
        
        info = detector.get_info()
        
        assert 'model_name' in info
        assert 'device' in info
        assert 'confidence_threshold' in info
        assert info['confidence_threshold'] == 0.45
        assert info['device'] == "cpu"
    
    def test_repr(self, mock_model_path):
        """Test string representation."""
        detector = RTDETRDetector(mock_model_path)
        repr_str = repr(detector)
        
        assert 'RTDETRDetector' in repr_str
        assert 'model_path' in repr_str


class TestRTDETRDetectorFactory:
    """Test detector factory."""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock model file."""
        model_path = tmp_path / "test_model.pt"
        model_path.touch()
        return str(model_path)
    
    def test_create_strict(self, mock_model_path):
        """Test creating strict detector."""
        detector = RTDETRDetectorFactory.create_strict(mock_model_path, device="cpu")
        assert detector.confidence_threshold == 0.30
    
    def test_create_balanced(self, mock_model_path):
        """Test creating balanced detector."""
        detector = RTDETRDetectorFactory.create_balanced(mock_model_path, device="cpu")
        assert detector.confidence_threshold == 0.45
    
    def test_create_precision(self, mock_model_path):
        """Test creating precision detector."""
        detector = RTDETRDetectorFactory.create_precision(mock_model_path, device="cpu")
        assert detector.confidence_threshold == 0.65
    
    def test_create_detector_convenience_function(self, mock_model_path):
        """Test convenience function."""
        detector = create_detector(mock_model_path, preset="balanced", device="cpu")
        assert detector.confidence_threshold == 0.45
        
        with pytest.raises(ValueError):
            create_detector(mock_model_path, preset="invalid")


class TestRTDETRDetectorIntegration:
    """Integration tests (requires actual model)."""
    
    @pytest.mark.skipif(
        not Path("models/rtdetr_nudity.pt").exists(),
        reason="Model file not found"
    )
    def test_detect_with_real_model(self):
        """Test detection with real model."""
        detector = RTDETRDetector(
            "models/rtdetr_nudity.pt",
            device="cpu"  # Use CPU for CI/CD
        )
        
        # Create dummy frames
        frames = np.random.randint(0, 255, (2, 640, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.detect(frames)
        
        assert len(detections) == 2  # One list per frame
        assert isinstance(detections[0], list)
    
    @pytest.mark.skipif(
        not Path("models/rtdetr_nudity.pt").exists(),
        reason="Model file not found"
    )
    def test_warmup_with_real_model(self):
        """Test warmup with real model."""
        detector = RTDETRDetector(
            "models/rtdetr_nudity.pt",
            device="cpu"
        )
        
        # Should not raise
        detector.warmup(num_iterations=1)