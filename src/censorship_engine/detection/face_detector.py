"""RetinaFace detector wrapper for face detection."""

import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path

try:
    from retinaface import RetinaFace
except ImportError:
    raise ImportError(
        "RetinaFace not installed. Run: pip install retina-face"
    )

from censorship_engine.core.interfaces import Detector
from censorship_engine.core.datatypes import Detection
from censorship_engine.core.exceptions import DetectionError

logger = logging.getLogger(__name__)


class FaceDetector(Detector):
    """
    RetinaFace detector for face detection and localization.
    
    RetinaFace automatically downloads and manages its own model weights.
    No need to provide model_path - it handles everything internally.
    
    Features:
    - High accuracy face detection
    - Works on various face angles and lighting
    - Batch processing support
    - Configurable confidence threshold
    - Automatic model weight management
    
    Example:
        >>> detector = FaceDetector(confidence_threshold=0.5)
        >>> detector.warmup()
        >>> frames = np.random.randint(0, 255, (4, 640, 640, 3), dtype=np.uint8)
        >>> detections = detector.detect(frames)
        >>> print(f"Found {len(detections[0])} faces in first frame")
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run on ('cpu' or 'cuda')
            
        Raises:
            ValueError: If parameters are invalid
            
        Note:
            RetinaFace automatically downloads model weights on first use.
            Weights are cached in ~/.retinaface/weights/
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Validate parameters
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        
        logger.info("Face detector initialized (RetinaFace)")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info("Model weights will be downloaded automatically on first use")
    
    def detect(self, frames: np.ndarray) -> List[List[Detection]]:
        """
        Detect faces in batch of frames.
        
        Args:
            frames: Batch of frames, shape (B, H, W, C) in BGR format
            
        Returns:
            List of detection lists, one per frame.
            Each detection list contains Detection objects for that frame.
            
        Raises:
            DetectionError: If detection fails
            ValueError: If frames shape is invalid
            
        Example:
            >>> frames = np.random.randint(0, 255, (4, 640, 640, 3), dtype=np.uint8)
            >>> detections = detector.detect(frames)
            >>> for frame_idx, frame_dets in enumerate(detections):
            ...     print(f"Frame {frame_idx}: {len(frame_dets)} faces")
        """
        # Validate input
        if not isinstance(frames, np.ndarray):
            raise ValueError(f"frames must be numpy array, got {type(frames)}")
        
        if frames.ndim != 4:
            raise ValueError(
                f"frames must be 4D array (B, H, W, C), got shape {frames.shape}"
            )
        
        batch_size, height, width, channels = frames.shape
        
        if channels != 3:
            raise ValueError(
                f"frames must have 3 channels (BGR), got {channels}"
            )
        
        try:
            all_detections = []
            
            # Process each frame (RetinaFace doesn't support batch processing)
            for frame_idx, frame in enumerate(frames):
                frame_detections = self._detect_single_frame(
                    frame, 
                    frame_idx,
                    height,
                    width
                )
                all_detections.append(frame_detections)
            
            # Log detection stats
            total_faces = sum(len(dets) for dets in all_detections)
            if total_faces > 0:
                avg_conf = np.mean([
                    d.confidence 
                    for dets in all_detections 
                    for d in dets
                ])
                logger.debug(
                    f"Detected {total_faces} faces across {batch_size} frames "
                    f"(avg confidence: {avg_conf:.3f})"
                )
            
            return all_detections
            
        except Exception as e:
            raise DetectionError(f"Face detection failed: {e}") from e
    
    def _detect_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_height: int,
        frame_width: int
    ) -> List[Detection]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Single frame (H, W, C) in BGR format
            frame_idx: Frame index in batch
            frame_height: Frame height
            frame_width: Frame width
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # RetinaFace expects RGB
        frame_rgb = frame[:, :, ::-1].copy()
        
        try:
            # Run detection using RetinaFace static method
            # Note: RetinaFace.detect_faces() is a static method, not an instance method
            faces = RetinaFace.detect_faces(
                frame_rgb,
                threshold=self.confidence_threshold,
                allow_upscaling=False
            )
            
            # Check if any faces detected
            if not isinstance(faces, dict):
                return detections
            
            # Process each detected face
            for face_id, face_data in faces.items():
                # Extract bbox
                bbox = face_data['facial_area']  # [x1, y1, x2, y2]
                confidence = face_data['score']
                
                # Ensure bbox is within frame boundaries
                x1 = max(0, min(int(bbox[0]), frame_width - 1))
                y1 = max(0, min(int(bbox[1]), frame_height - 1))
                x2 = max(0, min(int(bbox[2]), frame_width))
                y2 = max(0, min(int(bbox[3]), frame_height))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    logger.warning(
                        f"Invalid face bbox detected: ({x1}, {y1}, {x2}, {y2}), skipping"
                    )
                    continue
                
                # Create Detection object
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=1,  # Face class ID
                    class_name='face',
                    frame_id=frame_idx,
                    model_name='retinaface'
                )
                detections.append(detection)
                
        except Exception as e:
            logger.warning(f"Face detection failed on frame {frame_idx}: {e}")
        
        return detections
    
    def warmup(self, warmup_size: Tuple[int, int] = (640, 640), num_iterations: int = 3) -> None:
        """
        Run warmup inference to initialize model.
        
        This is especially important for RetinaFace because:
        - First call downloads model weights (if not cached)
        - First inference is slower due to initialization
        
        Args:
            warmup_size: Size of warmup frames (height, width)
            num_iterations: Number of warmup iterations
            
        Example:
            >>> detector.warmup()
            >>> # Now real inference will be faster
        """
        logger.info(f"Running warmup inference ({num_iterations} iterations)...")
        logger.info("Note: First run may download model weights (~100MB)")
        
        height, width = warmup_size
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            try:
                # This will download weights on first call
                RetinaFace.detect_faces(
                    dummy_frame,
                    threshold=self.confidence_threshold
                )
                logger.debug(f"Warmup iteration {i+1}/{num_iterations} complete")
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        logger.info("Warmup complete - model weights cached")
    
    def get_info(self) -> dict:
        """
        Get detector information.
        
        Returns:
            Dictionary with detector metadata
            
        Example:
            >>> info = detector.get_info()
            >>> print(f"Model: {info['model_name']}")
            >>> print(f"Threshold: {info['confidence_threshold']}")
        """
        return {
            'model_name': 'RetinaFace',
            'confidence_threshold': self.confidence_threshold,
            'device': self.device,
            'class_names': {1: 'face'},
            'num_classes': 1,
            'weights_location': '~/.retinaface/weights/ (auto-managed)',
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0-1)
            
        Raises:
            ValueError: If threshold is invalid
            
        Example:
            >>> detector.set_confidence_threshold(0.3)  # More sensitive
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        
        logger.info(f"Updating confidence threshold: {self.confidence_threshold} → {threshold}")
        self.confidence_threshold = threshold
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FaceDetector(model='RetinaFace', "
            f"confidence={self.confidence_threshold}, "
            f"device='{self.device}')"
        )


# Convenience function
def create_face_detector(
    preset: str = "balanced",
    device: str = "cpu"
) -> FaceDetector:
    """
    Create face detector with preset configuration.
    
    Args:
        preset: Configuration preset ('strict', 'balanced', 'precision')
        device: Device to use
        
    Returns:
        Configured FaceDetector
        
    Raises:
        ValueError: If preset is invalid
        
    Example:
        >>> detector = create_face_detector(preset="strict")
        >>> detector.warmup()
    """
    presets = {
        'strict': 0.40,      # Catch more faces
        'balanced': 0.50,    # Default
        'precision': 0.70,   # Only high-confidence faces
    }
    
    if preset not in presets:
        raise ValueError(
            f"Invalid preset '{preset}'. Choose from: {list(presets.keys())}"
        )
    
    return FaceDetector(
        confidence_threshold=presets[preset],
        device=device
    )