"""Video loading with batching and quality filtering."""
import sys
import cv2
import numpy as np
from typing import Iterator, Tuple, List, Optional
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"/"censorship_engine"/"core"))
from censorship_engine.core.exceptions import VideoLoadError

logger = logging.getLogger(__name__)


class VideoLoader:
    """
    Video loader with batch support and quality filtering.
    
    Features:
    - Batch frame loading for efficient GPU processing
    - Quality filtering (blur detection)
    - Frame skipping for faster processing
    - Automatic resource cleanup
    
    Example:
        >>> with VideoLoader("video.mp4", batch_size=16) as loader:
        ...     for frame_ids, frames_batch in loader:
        ...         # Process batch of 16 frames
        ...         detections = detector.detect(frames_batch)
    """
    
    def __init__(
        self,
        file_path: str,
        batch_size: int = 16,
        skip_frames: int = 0,
        quality_threshold: float = 100.0,
        enable_quality_filter: bool = True
    ):
        """
        Initialize video loader.
        
        Args:
            file_path: Path to video file
            batch_size: Number of frames per batch
            skip_frames: Process every (skip_frames + 1) frames (0 = all frames)
            quality_threshold: Laplacian variance threshold for blur detection
            enable_quality_filter: Enable/disable quality filtering
            
        Raises:
            VideoLoadError: If video cannot be opened
            FileNotFoundError: If video file doesn't exist
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.skip_frames = skip_frames
        self.quality_threshold = quality_threshold
        self.enable_quality_filter = enable_quality_filter
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # Open video
        self.cap = cv2.VideoCapture(str(file_path))
        if not self.cap.isOpened():
            raise VideoLoadError(f"Could not open video: {file_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Statistics
        self.frames_read = 0
        self.frames_skipped_quality = 0
        self.frames_skipped_stride = 0
        
        logger.info(f"Loaded video: {file_path}")
        logger.info(f"Properties: {self.width}x{self.height} @ {self.fps:.2f} FPS, {self.frame_count} frames")
        logger.info(f"Batch size: {batch_size}, Skip frames: {skip_frames}")
    
    def __iter__(self) -> Iterator[Tuple[List[int], np.ndarray]]:
        """
        Iterate over video frames in batches.
        
        Yields:
            Tuple of (frame_ids, frames_batch) where:
            - frame_ids: List of frame indices
            - frames_batch: np.ndarray of shape (B, H, W, C) in BGR format
        """
        batch_frames = []
        batch_ids = []
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            
            # End of video
            if not ret:
                # Yield final partial batch if exists
                if len(batch_frames) > 0:
                    logger.debug(f"Yielding final batch of {len(batch_frames)} frames")
                    yield batch_ids, np.stack(batch_frames)
                break
            
            # Skip frames based on stride
            if frame_id % (self.skip_frames + 1) != 0:
                self.frames_skipped_stride += 1
                frame_id += 1
                continue
            
            # Quality check
            if self.enable_quality_filter and not self._is_good_quality(frame):
                self.frames_skipped_quality += 1
                frame_id += 1
                continue
            
            # Add to batch
            batch_frames.append(frame)
            batch_ids.append(frame_id)
            self.frames_read += 1
            frame_id += 1
            
            # Yield when batch is full
            if len(batch_frames) == self.batch_size:
                yield batch_ids, np.stack(batch_frames)
                batch_frames = []
                batch_ids = []
        
        # Log statistics
        logger.info(f"Video loading complete:")
        logger.info(f"  Frames read: {self.frames_read}")
        logger.info(f"  Frames skipped (stride): {self.frames_skipped_stride}")
        logger.info(f"  Frames skipped (quality): {self.frames_skipped_quality}")
    
    def _is_good_quality(self, frame: np.ndarray) -> bool:
        """
        Check if frame is sharp enough (blur detection).
        
        Uses Laplacian variance method:
        - High variance = sharp edges = good quality
        - Low variance = blurry = skip frame
        
        Args:
            frame: BGR frame
            
        Returns:
            True if frame quality is acceptable
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var > self.quality_threshold
        except Exception as e:
            logger.warning(f"Quality check failed: {e}, accepting frame")
            return True
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by index.
        
        Args:
            frame_number: Frame index (0-based)
            
        Returns:
            Frame as np.ndarray or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def reset(self) -> None:
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frames_read = 0
        self.frames_skipped_quality = 0
        self.frames_skipped_stride = 0
    
    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.debug(f"Released video: {self.file_path}")
    
    def get_stats(self) -> dict:
        """
        Get loading statistics.
        
        Returns:
            Dictionary with loading stats
        """
        return {
            'file_path': str(self.file_path),
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.frame_count,
            'frames_read': self.frames_read,
            'frames_skipped_stride': self.frames_skipped_stride,
            'frames_skipped_quality': self.frames_skipped_quality,
            'batch_size': self.batch_size,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoLoader(file_path='{self.file_path}', "
            f"batch_size={self.batch_size}, "
            f"resolution={self.width}x{self.height})"
        )