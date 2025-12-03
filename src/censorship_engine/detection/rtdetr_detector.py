"""RT-DETR detector wrapper for nudity detection."""
import sys
import os
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path

try:
    from ultralytics import RTDETR
    import torch
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Run: pip install ultralytics torch"
    ) from e

from censorship_engine.core.interfaces import Detector
from censorship_engine.core.datatypes import Detection
from censorship_engine.core.exceptions import ModelLoadError, DetectionError

logger = logging.getLogger(__name__)


class RTDETRDetector(Detector):
    """
    RT-DETR detector for nudity detection.
    
    This detector uses the RT-DETR (Real-Time DEtection TRansformer) architecture
    for accurate nudity detection in video frames.
    
    Features:
    - Batch inference for performance
    - GPU acceleration support
    - Configurable confidence thresholds
    - Optional TensorRT optimization
    
    Example:
        >>> detector = RTDETRDetector(
        ...     model_path="models/rtdetr_nudity.pt",
        ...     confidence_threshold=0.45,
        ...     device="cuda:0"
        ... )
        >>> detector.warmup()
        >>> frames = np.random.randint(0, 255, (8, 640, 640, 3), dtype=np.uint8)
        >>> detections = detector.detect(frames)
        >>> print(f"Found {len(detections[0])} detections in first frame")
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        device: str = "cpu",
        use_fp16: bool = False,
        verbose: bool = False
    ):
        """
        Initialize RT-DETR detector.
        
        Args:
            model_path: Path to trained RT-DETR model (.pt file)
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            device: Device to run inference on ('cuda:0', 'cuda:1', 'cpu')
            use_fp16: Use FP16 precision (faster, slightly less accurate)
            verbose: Enable verbose logging
            
        Raises:
            ModelLoadError: If model file not found or loading fails
            ValueError: If parameters are invalid
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.use_fp16 = use_fp16
        self.verbose = verbose
        
        # Validate parameters
        self._validate_parameters()
        
        # Load model
        logger.info(f"Loading RT-DETR model from {self.model_path}")
        self.model = self._load_model()
        
        # Get class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'nudity'}
        
        logger.info(f"RT-DETR detector initialized on {self.device}")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(
                f"iou_threshold must be in [0, 1], got {self.iou_threshold}"
            )
        
        if self.max_detections < 1:
            raise ValueError(
                f"max_detections must be >= 1, got {self.max_detections}"
            )
        
        # Check CUDA availability
        if "cuda" in self.device and not torch.cuda.is_available():
            logger.warning(
                f"CUDA device '{self.device}' requested but CUDA not available. "
                "Falling back to CPU."
            )
            self.device = "cpu"
    
    def _load_model(self) -> RTDETR:
        """
        Load RT-DETR model.
        
        Returns:
            Loaded RT-DETR model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Load model
            model = RTDETR(str(self.model_path))
            
            # Move to device
            model.to(self.device)
            
            # Set to eval mode
            if hasattr(model, 'model'):
                model.model.eval()
            
            # Enable FP16 if requested and available
            if self.use_fp16 and self.device != "cpu":
                if hasattr(model, 'model'):
                    model.model.half()
                logger.info("FP16 precision enabled")
            
            return model
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {self.model_path}: {e}") from e
    
    def detect(self, frames: np.ndarray) -> List[List[Detection]]:
        """
        Detect nudity in batch of frames.
        
        Args:
            frames: Batch of frames, shape (B, H, W, C) in BGR format
            
        Returns:
            List of detection lists, one per frame.
            
        Raises:
            DetectionError: If detection fails
            ValueError: If frames shape is invalid
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
            # Convert 4D numpy array to list of 3D arrays
            # Ultralytics expects List[np.ndarray], not a single 4D batch
            frames_list = list(frames)
            
            # Run inference
            results = self.model.predict(
                frames_list,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=self.verbose,
                device=self.device
            )
            
            # Convert results to Detection objects
            all_detections = []
            
            for frame_idx, result in enumerate(results):
                frame_detections = self._convert_result_to_detections(
                    result, 
                    frame_idx,
                    height,
                    width
                )
                all_detections.append(frame_detections)
            
            # Log detection stats
            total_dets = sum(len(dets) for dets in all_detections)
            if total_dets > 0:
                avg_conf = np.mean([
                    d.confidence 
                    for dets in all_detections 
                    for d in dets
                ])
                logger.debug(
                    f"Detected {total_dets} objects across {batch_size} frames "
                    f"(avg confidence: {avg_conf:.3f})"
                )
            
            return all_detections
            
        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e
    
    def _convert_result_to_detections(
        self, 
        result, 
        frame_idx: int,
        frame_height: int,
        frame_width: int
    ) -> List[Detection]:
        """
        Convert Ultralytics result to Detection objects.
        
        Args:
            result: Ultralytics detection result
            frame_idx: Frame index in batch
            frame_height: Original frame height
            frame_width: Original frame width
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Check if any detections exist
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Create Detection objects
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Ensure bbox is within frame boundaries
            x1, y1, x2, y2 = box
            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width))
            y2 = max(0, min(int(y2), frame_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    f"Invalid bbox detected: ({x1}, {y1}, {x2}, {y2}), skipping"
                )
                continue
            
            # Create Detection object
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                frame_id=frame_idx,
                model_name="rtdetr"
            )
            detections.append(detection)
        
        return detections
    
    def warmup(self, warmup_size: Tuple[int, int] = (640, 640), num_iterations: int = 3) -> None:
        """
        Run warmup inference to initialize model.
        
        The first inference is often slower due to model initialization,
        CUDA kernel compilation, etc. This method runs dummy inferences
        to warm up the model.
        
        Args:
            warmup_size: Size of warmup frames (height, width)
            num_iterations: Number of warmup iterations
            
        Example:
            >>> detector.warmup()
            >>> # Now real inference will be faster
        """
        logger.info(f"Running warmup inference ({num_iterations} iterations)...")
        
        height, width = warmup_size
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            try:
                self.model.predict(
                    dummy_frame,
                    conf=self.confidence_threshold,
                    verbose=False
                )
                logger.debug(f"Warmup iteration {i+1}/{num_iterations} complete")
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        logger.info("Warmup complete")
    
    def get_info(self) -> dict:
        """
        Get detector information.
        
        Returns:
            Dictionary with detector metadata
            
        Example:
            >>> info = detector.get_info()
            >>> print(f"Model: {info['model_name']}")
            >>> print(f"Device: {info['device']}")
        """
        info = {
            'model_name': 'RT-DETR',
            'model_path': str(self.model_path),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections,
            'use_fp16': self.use_fp16,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
        }
        
        # Add GPU info if available
        if torch.cuda.is_available() and "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            info['gpu_name'] = torch.cuda.get_device_name(device_idx)
            info['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated(device_idx) / 1e9:.2f} GB"
            info['gpu_memory_reserved'] = f"{torch.cuda.memory_reserved(device_idx) / 1e9:.2f} GB"
        
        return info
    
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
            f"RTDETRDetector(model_path='{self.model_path}', "
            f"confidence={self.confidence_threshold}, "
            f"device='{self.device}')"
        )


class RTDETRDetectorFactory:
    """Factory for creating RT-DETR detector instances with presets."""
    
    @staticmethod
    def create_strict(model_path: str, device: str = "cuda:0") -> RTDETRDetector:
        """
        Create detector with strict settings (high recall).
        
        Args:
            model_path: Path to model file
            device: Device to use
            
        Returns:
            Configured RTDETRDetector
        """
        return RTDETRDetector(
            model_path=model_path,
            confidence_threshold=0.30,  # Lower threshold = more detections
            iou_threshold=0.45,
            device=device
        )
    
    @staticmethod
    def create_balanced(model_path: str, device: str = "cuda:0") -> RTDETRDetector:
        """
        Create detector with balanced settings (default).
        
        Args:
            model_path: Path to model file
            device: Device to use
            
        Returns:
            Configured RTDETRDetector
        """
        return RTDETRDetector(
            model_path=model_path,
            confidence_threshold=0.45,
            iou_threshold=0.50,
            device=device
        )
    
    @staticmethod
    def create_precision(model_path: str, device: str = "cuda:0") -> RTDETRDetector:
        """
        Create detector with precision settings (high precision).
        
        Args:
            model_path: Path to model file
            device: Device to use
            
        Returns:
            Configured RTDETRDetector
        """
        return RTDETRDetector(
            model_path=model_path,
            confidence_threshold=0.65,  # Higher threshold = fewer false positives
            iou_threshold=0.55,
            device=device
        )


# Convenience function for quick usage
def create_detector(
    model_path: str,
    preset: str = "balanced",
    device: str = "cuda:0"
) -> RTDETRDetector:
    """
    Create RT-DETR detector with preset configuration.
    
    Args:
        model_path: Path to trained model
        preset: Configuration preset ('strict', 'balanced', 'precision')
        device: Device to use
        
    Returns:
        Configured RTDETRDetector
        
    Raises:
        ValueError: If preset is invalid
        
    Example:
        >>> detector = create_detector("models/rtdetr.pt", preset="strict")
        >>> detector.warmup()
    """
    factory = RTDETRDetectorFactory()
    
    presets = {
        'strict': factory.create_strict,
        'balanced': factory.create_balanced,
        'precision': factory.create_precision,
    }
    
    if preset not in presets:
        raise ValueError(
            f"Invalid preset '{preset}'. Choose from: {list(presets.keys())}"
        )
    
    return presets[preset](model_path, device)