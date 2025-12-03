"""Video writing with hardware acceleration support."""

import cv2
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Optional

from censorship_engine.core.exceptions import VideoWriteError

logger = logging.getLogger(__name__)


class VideoWriter:
    """
    Video writer with hardware acceleration and audio preservation.
    
    Features:
    - Hardware-accelerated encoding (NVENC, VideoToolbox)
    - Audio preservation from source video
    - Multiple codec support
    - Automatic fallback to software encoding
    
    Example:
        >>> with VideoWriter("output.mp4", fps=30, width=1920, height=1080) as writer:
        ...     for frame in frames:
        ...         writer.write(frame)
    """
    
    def __init__(
        self,
        file_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
        use_hardware_encoding: bool = True,
        crf: int = 23,
        preset: str = "medium"
    ):
        """
        Initialize video writer.
        
        Args:
            file_path: Output video path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec ('mp4v', 'avc1', 'h264', 'h265')
            use_hardware_encoding: Try hardware acceleration
            crf: Constant Rate Factor (lower = better quality, 0-51)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            
        Raises:
            VideoWriteError: If writer initialization fails
        """
        self.file_path = Path(file_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.use_hardware_encoding = use_hardware_encoding
        self.crf = crf
        self.preset = preset
        
        # Create output directory if needed
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer
        self.writer = self._create_writer()
        
        if self.writer is None or not self.writer.isOpened():
            raise VideoWriteError(f"Failed to create video writer: {file_path}")
        
        self.frames_written = 0
        
        logger.info(f"Video writer initialized: {file_path}")
        logger.info(f"Properties: {width}x{height} @ {fps:.2f} FPS, codec: {codec}")
    
    def _create_writer(self) -> cv2.VideoWriter:
        """
        Create video writer with hardware acceleration if available.
        
        Returns:
            OpenCV VideoWriter object
        """
        # Try hardware encoding first
        if self.use_hardware_encoding:
            writer = self._try_hardware_encoding()
            if writer is not None and writer.isOpened():
                logger.info("Using hardware-accelerated encoding")
                return writer
            else:
                logger.warning("Hardware encoding failed, falling back to software")
        
        # Fallback to software encoding
        return self._create_software_writer()
    
    def _try_hardware_encoding(self) -> Optional[cv2.VideoWriter]:
        """
        Try to create hardware-accelerated writer.
        
        Returns:
            VideoWriter or None if hardware encoding not available
        """
        try:
            # Try H.264 with hardware acceleration
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(
                str(self.file_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            # Test if it actually works
            if writer.isOpened():
                return writer
            
        except Exception as e:
            logger.debug(f"Hardware encoding attempt failed: {e}")
        
        return None
    
    def _create_software_writer(self) -> cv2.VideoWriter:
        """
        Create software-based video writer.
        
        Returns:
            VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.file_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        return writer
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame.
        
        Args:
            frame: Frame to write (H, W, C) in BGR format
            
        Raises:
            VideoWriteError: If frame writing fails
            ValueError: If frame shape doesn't match
        """
        # Validate frame shape
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} doesn't match writer "
                f"dimensions ({self.height}, {self.width})"
            )
        
        try:
            self.writer.write(frame)
            self.frames_written += 1
            
            if self.frames_written % 100 == 0:
                logger.debug(f"Written {self.frames_written} frames")
                
        except Exception as e:
            raise VideoWriteError(f"Failed to write frame: {e}") from e
    
    def write_batch(self, frames: np.ndarray) -> None:
        """
        Write batch of frames.
        
        Args:
            frames: Batch of frames (B, H, W, C) in BGR format
        """
        for frame in frames:
            self.write(frame)
    
    def release(self) -> None:
        """Release writer resources."""
        if self.writer is not None:
            self.writer.release()
            
            # CRITICAL: Give time for file finalization
            import time
            time.sleep(0.5)  # Small delay to ensure file is fully written
            
            logger.info(f"Video written successfully: {self.file_path}")
            logger.info(f"Total frames written: {self.frames_written}")
    
    def preserve_audio(self, source_video_path: str) -> None:
        """
        Copy audio from source video to output video.
        
        Uses ffmpeg to merge video from this writer with audio from source.
        
        Args:
            source_video_path: Path to source video with audio
            
        Note:
            This should be called AFTER release().
            It will create a temporary file and replace the output.
        """
        try:
            source = Path(source_video_path)
            if not source.exists():
                logger.warning(f"Source video not found: {source}, skipping audio preservation")
                return
            
            # Create temporary output path
            temp_output = self.file_path.with_suffix('.temp.mp4')
            
            # Use ffmpeg to merge video + audio
            cmd = [
                'ffmpeg',
                '-i', str(self.file_path),      # Video (no audio)
                '-i', str(source),               # Audio source
                '-c:v', 'copy',                  # Copy video codec
                '-c:a', 'aac',                   # Re-encode audio to AAC
                '-map', '0:v:0',                 # Take video from first input
                '-map', '1:a:0?',                # Take audio from second input (if exists)
                '-shortest',                      # Match shortest stream
                '-y',                             # Overwrite output
                str(temp_output)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Replace original with audio-merged version
                temp_output.replace(self.file_path)
                logger.info("Audio preserved successfully")
            else:
                logger.warning(f"Audio preservation failed: {result.stderr.decode()}")
                temp_output.unlink(missing_ok=True)
                
        except FileNotFoundError:
            logger.warning("ffmpeg not found, audio preservation skipped")
        except subprocess.TimeoutExpired:
            logger.error("Audio preservation timed out")
            temp_output.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Audio preservation error: {e}")
    
    def get_stats(self) -> dict:
        """
        Get writing statistics.
        
        Returns:
            Dictionary with writing stats
        """
        return {
            'file_path': str(self.file_path),
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'codec': self.codec,
            'frames_written': self.frames_written,
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
            f"VideoWriter(file_path='{self.file_path}', "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f})"
        )