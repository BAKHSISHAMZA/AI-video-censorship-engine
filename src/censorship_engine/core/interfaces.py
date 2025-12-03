"""Abstract base classes for the censorship engine."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .datatypes import Detection, Track


class Detector(ABC):
    """Base class for all detection models."""
    
    @abstractmethod
    def detect(self, frames: np.ndarray) -> List[List[Detection]]:
        """
        Detect objects in a batch of frames.
        
        Args:
            frames: (B, H, W, C) array of BGR frames
            
        Returns:
            List of detections per frame, shape: [batch_size][num_detections]
        """
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """
        Run warmup inference to initialize model.
        
        First inference is often slower due to model initialization.
        This method runs a dummy inference to warm up the model.
        """
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """
        Get detector information.
        
        Returns:
            Dictionary with model info (name, version, parameters, etc.)
        """
        pass


class Tracker(ABC):
    """Base class for all tracking algorithms."""
    
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: New detections from current frame
            frame: Current frame (for feature extraction if needed)
            
        Returns:
            List of active tracks
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state (clear all tracks)."""
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        pass
    
    @abstractmethod
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (active and inactive)."""
        pass


class Renderer(ABC):
    """Base class for all rendering strategies."""
    
    @abstractmethod
    def render(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Apply censorship to frame based on tracks.
        
        Args:
            frame: Original frame (H, W, C) BGR
            tracks: Active tracks to censor
            
        Returns:
            Censored frame (same shape as input)
        """
        pass
    
    @abstractmethod
    def render_bbox(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Apply censorship to a specific bounding box.
        
        Args:
            frame: Original frame
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            Frame with censored region
        """
        pass


class MetricsCollector(ABC):
    """Base class for metrics collection."""
    
    @abstractmethod
    def start_stage(self, stage_name: str) -> None:
        """Start timing a pipeline stage."""
        pass
    
    @abstractmethod
    def end_stage(self, stage_name: str) -> None:
        """End timing a pipeline stage."""
        pass
    
    @abstractmethod
    def record_detection(self, detection: Detection) -> None:
        """Record a detection."""
        pass
    
    @abstractmethod
    def record_track(self, track: Track) -> None:
        """Record a track."""
        pass
    
    @abstractmethod
    def get_report(self) -> dict:
        """Get performance report."""
        pass