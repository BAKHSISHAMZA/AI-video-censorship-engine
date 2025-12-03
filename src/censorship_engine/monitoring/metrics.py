"""
Performance metrics collection for the censorship pipeline.

Tracks timing, detection counts, and other performance statistics.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from censorship_engine.core.interfaces import MetricsCollector
from censorship_engine.core.datatypes import Detection, Track

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    
    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    
    @property
    def mean_time_ms(self) -> float:
        """Average time per call."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0
    
    def update(self, time_ms: float):
        """Update metrics with new timing."""
        self.call_count += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'call_count': self.call_count,
            'total_time_ms': self.total_time_ms,
            'mean_ms': self.mean_time_ms,
            'min_ms': self.min_time_ms if self.min_time_ms != float('inf') else 0.0,
            'max_ms': self.max_time_ms,
        }


class SimpleMetricsCollector(MetricsCollector):
    """
    Simple metrics collector for pipeline performance tracking.
    
    Tracks:
    - Stage timings (detection, tracking, rendering)
    - Detection counts by class
    - Track statistics
    - Memory usage (if available)
    
    Example:
        >>> metrics = SimpleMetricsCollector()
        >>> with metrics.measure('detection'):
        ...     # detection code
        ...     pass
        >>> report = metrics.get_report()
        >>> print(f"Detection took {report['stage_timings']['detection']['mean_ms']:.1f}ms")
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        # Stage timings
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.active_stages: Dict[str, float] = {}  # stage_name -> start_time
        
        # Detection statistics
        self.detection_counts: Dict[str, int] = defaultdict(int)  # class_name -> count
        self.total_detections = 0
        self.detection_confidences: List[float] = []
        
        # Track statistics
        self.track_ids_seen: set = set()
        self.track_ages: List[int] = []
        self.track_hits: List[int] = []
        
        # Overall timing
        self.pipeline_start_time: Optional[float] = None
        self.pipeline_end_time: Optional[float] = None
        
        logger.debug("Metrics collector initialized")
    
    def start_stage(self, stage_name: str) -> None:
        """
        Start timing a pipeline stage.
        
        Args:
            stage_name: Name of the stage (e.g., 'detection', 'tracking')
        """
        if stage_name in self.active_stages:
            logger.warning(f"Stage '{stage_name}' already started, overwriting")
        
        self.active_stages[stage_name] = time.time()
    
    def end_stage(self, stage_name: str) -> None:
        """
        End timing a pipeline stage.
        
        Args:
            stage_name: Name of the stage
        """
        if stage_name not in self.active_stages:
            logger.warning(f"Stage '{stage_name}' was not started")
            return
        
        # Calculate elapsed time
        start_time = self.active_stages.pop(stage_name)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update stage metrics
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = StageMetrics(name=stage_name)
        
        self.stage_metrics[stage_name].update(elapsed_ms)
        
        logger.debug(f"Stage '{stage_name}' took {elapsed_ms:.2f}ms")
    
    def measure(self, stage_name: str):
        """
        Context manager for measuring stage timing.
        
        Args:
            stage_name: Name of the stage
            
        Example:
            >>> with metrics.measure('detection'):
            ...     detections = detector.detect(frame)
        """
        return _StageTimer(self, stage_name)
    
    def record_detection(self, detection: Detection) -> None:
        """
        Record a detection.
        
        Args:
            detection: Detection object to record
        """
        self.total_detections += 1
        self.detection_counts[detection.class_name] += 1
        self.detection_confidences.append(detection.confidence)
        
        logger.debug(f"Recorded detection: {detection.class_name} ({detection.confidence:.2f})")
    
    def record_track(self, track: Track) -> None:
        """
        Record a track.
        
        Args:
            track: Track object to record
        """
        # Track unique IDs
        if hasattr(track, 'track_id'):
            self.track_ids_seen.add(track.track_id)
        
        # Record age
        if hasattr(track, 'age'):
            self.track_ages.append(track.age)
        
        # Record hits
        if hasattr(track, 'hits'):
            self.track_hits.append(track.hits)
        
        logger.debug(f"Recorded track: ID={getattr(track, 'track_id', 'unknown')}")
    
    def start_pipeline(self) -> None:
        """Mark the start of pipeline processing."""
        self.pipeline_start_time = time.time()
        logger.debug("Pipeline timing started")
    
    def end_pipeline(self) -> None:
        """Mark the end of pipeline processing."""
        self.pipeline_end_time = time.time()
        logger.debug("Pipeline timing ended")
    
    def get_report(self) -> dict:
        """
        Get comprehensive performance report.
        
        Returns:
            Dictionary with all collected metrics
            
        Example:
            >>> report = metrics.get_report()
            >>> print(f"Total detections: {report['detections']['total']}")
        """
        # Calculate pipeline duration
        pipeline_duration_s = 0.0
        if self.pipeline_start_time and self.pipeline_end_time:
            pipeline_duration_s = self.pipeline_end_time - self.pipeline_start_time
        
        # Stage timings
        stage_timings = {}
        for stage_name, metrics in self.stage_metrics.items():
            stage_timings[stage_name] = metrics.to_dict()
        
        # Detection statistics
        detection_stats = {
            'total': self.total_detections,
            'by_class': dict(self.detection_counts),
            'avg_confidence': sum(self.detection_confidences) / len(self.detection_confidences) 
                if self.detection_confidences else 0.0,
            'min_confidence': min(self.detection_confidences) if self.detection_confidences else 0.0,
            'max_confidence': max(self.detection_confidences) if self.detection_confidences else 0.0,
        }
        
        # Track statistics
        track_stats = {
            'unique_tracks': len(self.track_ids_seen),
            'avg_age': sum(self.track_ages) / len(self.track_ages) if self.track_ages else 0.0,
            'max_age': max(self.track_ages) if self.track_ages else 0,
            'avg_hits': sum(self.track_hits) / len(self.track_hits) if self.track_hits else 0.0,
        }
        
        # Memory usage (optional)
        memory_stats = self._get_memory_stats()
        
        return {
            'pipeline_duration_seconds': pipeline_duration_s,
            'stage_timings': stage_timings,
            'detections': detection_stats,
            'tracks': track_stats,
            'memory': memory_stats,
        }
    
    def _get_memory_stats(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory stats (if available)
        """
        stats = {}
        
        # Try to get GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    stats[f'gpu_{i}_allocated_mb'] = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    stats[f'gpu_{i}_reserved_mb'] = torch.cuda.memory_reserved(i) / (1024 ** 2)
                    stats[f'gpu_{i}_max_allocated_mb'] = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        except Exception as e:
            logger.debug(f"Could not get GPU memory stats: {e}")
        
        # Try to get CPU memory
        try:
            import psutil
            process = psutil.Process()
            stats['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
            stats['cpu_memory_percent'] = process.memory_percent()
        except Exception as e:
            logger.debug(f"Could not get CPU memory stats: {e}")
        
        return stats
    
    def print_summary(self) -> None:
        """Print a formatted summary of metrics."""
        report = self.get_report()
        
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 70)
        
        # Pipeline duration
        if report['pipeline_duration_seconds'] > 0:
            print(f"\n⏱️  Total Duration: {report['pipeline_duration_seconds']:.2f}s")
        
        # Stage timings
        print(f"\n📊 Stage Timings:")
        for stage_name, metrics in report['stage_timings'].items():
            print(f"   {stage_name.capitalize()}:")
            print(f"      Calls: {metrics['call_count']}")
            print(f"      Mean: {metrics['mean_ms']:.1f}ms")
            print(f"      Min: {metrics['min_ms']:.1f}ms")
            print(f"      Max: {metrics['max_ms']:.1f}ms")
            print(f"      Total: {metrics['total_time_ms']:.1f}ms")
        
        # Detections
        det_stats = report['detections']
        print(f"\n🎯 Detections:")
        print(f"   Total: {det_stats['total']}")
        print(f"   Avg confidence: {det_stats['avg_confidence']:.3f}")
        if det_stats['by_class']:
            print(f"   By class:")
            for class_name, count in det_stats['by_class'].items():
                print(f"      {class_name}: {count}")
        
        # Tracks
        track_stats = report['tracks']
        if track_stats['unique_tracks'] > 0:
            print(f"\n🔗 Tracks:")
            print(f"   Unique tracks: {track_stats['unique_tracks']}")
            print(f"   Avg age: {track_stats['avg_age']:.1f} frames")
            print(f"   Max age: {track_stats['max_age']} frames")
            print(f"   Avg hits: {track_stats['avg_hits']:.1f}")
        
        # Memory
        mem_stats = report['memory']
        if mem_stats:
            print(f"\n💾 Memory Usage:")
            for key, value in mem_stats.items():
                if 'gpu' in key:
                    print(f"   {key}: {value:.1f} MB")
                elif 'cpu' in key and 'percent' not in key:
                    print(f"   {key}: {value:.1f} MB")
        
        print("\n" + "=" * 70)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.stage_metrics.clear()
        self.active_stages.clear()
        self.detection_counts.clear()
        self.total_detections = 0
        self.detection_confidences.clear()
        self.track_ids_seen.clear()
        self.track_ages.clear()
        self.track_hits.clear()
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        logger.debug("Metrics reset")
    
    def export_csv(self, filepath: str) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        import csv
        
        report = self.get_report()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric Type', 'Metric Name', 'Value'])
            
            # Pipeline duration
            writer.writerow(['Pipeline', 'Duration (s)', report['pipeline_duration_seconds']])
            
            # Stage timings
            for stage_name, metrics in report['stage_timings'].items():
                writer.writerow(['Stage', f'{stage_name}_calls', metrics['call_count']])
                writer.writerow(['Stage', f'{stage_name}_mean_ms', metrics['mean_ms']])
                writer.writerow(['Stage', f'{stage_name}_min_ms', metrics['min_ms']])
                writer.writerow(['Stage', f'{stage_name}_max_ms', metrics['max_ms']])
            
            # Detections
            det_stats = report['detections']
            writer.writerow(['Detection', 'Total', det_stats['total']])
            writer.writerow(['Detection', 'Avg Confidence', det_stats['avg_confidence']])
            for class_name, count in det_stats['by_class'].items():
                writer.writerow(['Detection', f'Class_{class_name}', count])
            
            # Tracks
            track_stats = report['tracks']
            writer.writerow(['Track', 'Unique Tracks', track_stats['unique_tracks']])
            writer.writerow(['Track', 'Avg Age', track_stats['avg_age']])
            writer.writerow(['Track', 'Max Age', track_stats['max_age']])
        
        logger.info(f"Metrics exported to CSV: {filepath}")


class _StageTimer:
    """Context manager for measuring stage timing."""
    
    def __init__(self, collector: SimpleMetricsCollector, stage_name: str):
        self.collector = collector
        self.stage_name = stage_name
    
    def __enter__(self):
        self.collector.start_stage(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.collector.end_stage(self.stage_name)
        return False  # Don't suppress exceptions


class DetailedMetricsCollector(SimpleMetricsCollector):
    """
    Extended metrics collector with additional tracking.
    
    Adds:
    - Per-frame timing breakdown
    - Batch size tracking
    - Frame quality metrics
    - More detailed memory tracking
    """
    
    def __init__(self):
        super().__init__()
        self.frame_timings: List[Dict[str, float]] = []
        self.batch_sizes: List[int] = []
        self.frame_qualities: List[float] = []
    
    def record_frame_timing(self, frame_id: int, timings: Dict[str, float]) -> None:
        """
        Record per-frame timing breakdown.
        
        Args:
            frame_id: Frame index
            timings: Dictionary of stage_name -> time_ms
        """
        self.frame_timings.append({
            'frame_id': frame_id,
            **timings
        })
    
    def record_batch_size(self, batch_size: int) -> None:
        """Record batch size used."""
        self.batch_sizes.append(batch_size)
    
    def record_frame_quality(self, quality_score: float) -> None:
        """Record frame quality score (e.g., blur metric)."""
        self.frame_qualities.append(quality_score)
    
    def get_report(self) -> dict:
        """Get extended report with additional metrics."""
        report = super().get_report()
        
        # Add extended metrics
        report['batch_statistics'] = {
            'avg_batch_size': sum(self.batch_sizes) / len(self.batch_sizes) 
                if self.batch_sizes else 0.0,
            'min_batch_size': min(self.batch_sizes) if self.batch_sizes else 0,
            'max_batch_size': max(self.batch_sizes) if self.batch_sizes else 0,
        }
        
        report['frame_quality'] = {
            'avg_quality': sum(self.frame_qualities) / len(self.frame_qualities) 
                if self.frame_qualities else 0.0,
            'min_quality': min(self.frame_qualities) if self.frame_qualities else 0.0,
            'max_quality': max(self.frame_qualities) if self.frame_qualities else 0.0,
        }
        
        return report


# Convenience function
def create_metrics_collector(detailed: bool = False) -> MetricsCollector:
    """
    Factory function to create metrics collector.
    
    Args:
        detailed: If True, return DetailedMetricsCollector with extra features
        
    Returns:
        MetricsCollector instance
        
    Example:
        >>> metrics = create_metrics_collector(detailed=True)
    """
    if detailed:
        return DetailedMetricsCollector()
    else:
        return SimpleMetricsCollector()