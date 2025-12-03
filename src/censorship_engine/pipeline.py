"""
Main censorship pipeline - UPDATED to use enhanced renderer.

Changes from original:
1. Proper per-class strategy support
2. Enhanced strategy options building
3. Better error handling
4. Fixed blackbox rendering
"""

from __future__ import annotations
from typing import Optional, Dict, List
import logging
from pathlib import Path
import time
import json

import numpy as np

from censorship_engine.core.datatypes import (
    ProcessingConfig,
    ProcessingResult,
    Detection,
    CensorshipMethod
)
from censorship_engine.core.exceptions import (
    CensorshipEngineError,
    VideoLoadError,
    DetectionError,
    VideoWriteError
)
from censorship_engine.video.loader import VideoLoader
from censorship_engine.video.writer import VideoWriter
from censorship_engine.detection.rtdetr_detector import RTDETRDetector
from censorship_engine.detection.face_detector import FaceDetector
from censorship_engine.fusion.detection_fuser import DetectionFuser
from censorship_engine.tracking.deep_sort import DeepSortTracker, Track
from censorship_engine.rendering.censorship import CensorshipRenderer
from censorship_engine.monitoring.metrics import SimpleMetricsCollector
from censorship_engine.conversion.onnx_runtime import OptimizedRTDETRONNXDetector , OptimizedFaceONNXDetector

logger = logging.getLogger(__name__)


class CensorshipPipeline:
    """
    Production-ready video censorship pipeline with per-class strategies.
    
    Example:
        >>> config = ProcessingConfig(
        ...     class_strategies={
        ...         'face': 'pixelate',
        ...         'breast': 'blur',
        ...         'genitals': 'blackbox'
        ...     },
        ...     pixelate_block_size=20,
        ...     blur_kernel_size=51,
        ...     blackbox_color=(0, 0, 0),
        ...     blackbox_opacity=1.0
        ... )
        >>> pipeline = CensorshipPipeline(config)
        >>> result = pipeline.process_video("input.mp4", "output.mp4")
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        
        logger.info("=" * 70)
        logger.info("Initializing Censorship Pipeline")
        logger.info("=" * 70)
        
        # Initialize detectors
        logger.info("Loading detection models...")
        self.nudity_detector = OptimizedRTDETRONNXDetector(
            model_path=config.rtdetr_model_path,
            confidence_threshold=config.nudity_confidence_threshold,
            device=config.device,
            input_size=640  # Add this parameter explicitly
        )
        
        self.face_detector = OptimizedFaceONNXDetector(
            confidence_threshold=config.face_confidence_threshold,
            device="cpu"
        )
        
        # Fusion
        self.fuser = DetectionFuser(
            iou_threshold=config.iou_threshold
        )
        
        # Tracker
        self.tracker = DeepSortTracker(
            max_age=config.max_track_age,
            min_hits=config.min_track_hits,
            iou_threshold=config.iou_threshold,
            use_cosine=False
        )
        
        # Build strategy configuration
        strategy_map = self._build_strategy_map()
        strategy_options = self._build_strategy_options()
        
        logger.info("\nCensorship Strategies:")
        logger.info(f"  Default: {config.default_censorship_method.value}")
        if strategy_map:
            for cls, strat in strategy_map.items():
                logger.info(f"  {cls}: {strat}")
        
        # Initialize renderer with enhanced configuration
        self.renderer = CensorshipRenderer(
            default_strategy=config.default_censorship_method.value,
            strategy_map=strategy_map,
            strategy_options=strategy_options,
            merge_iou=config.iou_threshold,
            blend_alpha=config.smoothing_alpha if config.temporal_smoothing else 0.0
        )
        
        # Metrics
        self.metrics = SimpleMetricsCollector()
        
        # Warmup
        logger.info("\nWarming up models...")
        dummy_batch = np.zeros((1, 640, 640, 3), dtype=np.uint8)
        self.nudity_detector.warmup(dummy_batch,num_iterations=1)
        # In pipeline.py __init__ method, around line 119:
        try:
            self.face_detector.warmup(dummy_batch, num_iterations=1)
        except Exception as e:
            logger.warning(f"Face detector warmup failed: {e}")
                
        logger.info("=" * 70)
        logger.info("✓ Pipeline initialized successfully")
        logger.info("=" * 70)
    
    def _build_strategy_map(self) -> Dict[str, str]:
        """
        Build per-class strategy mapping from config.
        
        Returns:
            Dictionary mapping class_name -> strategy_name
        """
        if self.config.class_strategies:
            # Use user-provided strategies
            return {
                k.lower(): v.lower() 
                for k, v in self.config.class_strategies.items()
            }
        
        # Default: use config's default for all
        return {}
    
    def _build_strategy_options(self) -> Dict[str, dict]:
        """
        Build strategy options from config.
        
        Returns:
            Dictionary mapping strategy_name -> options dict
        """
        options = {}
        
        # Blur options
        options['blur'] = {
            'kernel_size': self.config.blur_kernel_size,
            'intensity': self.config.blur_intensity
        }
        options['gaussian'] = options['blur']  # Alias
        
        # Motion blur options
        options['motion_blur'] = {
            'kernel_size': getattr(self.config, 'motion_blur_kernel_size', 25),
            'angle': getattr(self.config, 'motion_blur_angle', 0.0),
            'intensity': getattr(self.config, 'motion_blur_intensity', 1.0)
        }
        options['motion'] = options['motion_blur']  # Alias
        
        # Pixelate options
        options['pixelate'] = {
            'pixel_block': self.config.pixelate_block_size,
            'intensity': self.config.pixelate_intensity
        }
        options['pixel'] = options['pixelate']  # Alias
        
        # BlackBox options
        blackbox_opts = {
            'color': self.config.blackbox_color,
            'opacity': self.config.blackbox_opacity,
            'rounded_corners': self.config.blackbox_rounded_corners,
            'intensity': 1.0  # Full intensity by default
        }
        options['blackbox'] = blackbox_opts
        options['black'] = blackbox_opts  # Alias
        options['box'] = blackbox_opts  # Alias
        options['blackout'] = blackbox_opts  # Alias
        
        # Emoji options
        options['emoji'] = {
            'scale': getattr(self.config, 'emoji_scale', 1.0),
            'intensity': getattr(self.config, 'emoji_intensity', 1.0)
        }
        
        return options
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Process video file with censorship.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Optional callback(current, total)
        
        Returns:
            ProcessingResult with statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise VideoLoadError(f"Input video not found: {input_path}")
        
        logger.info(f"\nProcessing: {input_path}")
        logger.info(f"Output: {output_path}")
        
        result = ProcessingResult(
            input_path=str(input_path),
            output_path=str(output_path),
            config=self.config
        )
        
        start_time = time.time()
        
        try:
            self._process_video_internal(
                input_path,
                output_path,
                result,
                progress_callback
            )
            
            # Calculate metrics
            result.processing_time_seconds = time.time() - start_time
            result.fps = result.frames_processed / result.processing_time_seconds if result.processing_time_seconds > 0 else 0
            
            metrics_report = self.metrics.get_report()
            result.avg_detection_time_ms = metrics_report['stage_timings'].get('detection', {}).get('mean_ms', 0)
            result.avg_tracking_time_ms = metrics_report['stage_timings'].get('tracking', {}).get('mean_ms', 0)
            result.avg_rendering_time_ms = metrics_report['stage_timings'].get('rendering', {}).get('mean_ms', 0)
            
            result.total_tracks = len(self.tracker.tracks)
            
            # Export metadata
            if self.config.export_metadata:
                result.metadata_path = self._export_metadata(result, metrics_report)
            
            # Preserve audio
            if self.config.preserve_audio:
                self._preserve_audio(input_path, output_path)
            
            # Log results
            self._log_results(result)
            
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise CensorshipEngineError(f"Video processing failed: {e}") from e
    
    def _process_video_internal(
        self,
        input_path: Path,
        output_path: Path,
        result: ProcessingResult,
        progress_callback: Optional[callable]
    ):
        """Internal video processing loop."""
        
        with VideoLoader(
            str(input_path),
            batch_size=self.config.batch_size,
            skip_frames=self.config.skip_frames,
            quality_threshold=self.config.blur_quality_threshold,
            enable_quality_filter=False
        ) as loader:
            
            result.total_frames = loader.frame_count
            logger.info(f"Total frames: {result.total_frames}")
            
            with VideoWriter(
                str(output_path),
                fps=loader.fps,
                width=loader.width,
                height=loader.height,
                use_hardware_encoding=self.config.use_nvenc
            ) as writer:
                
                prev_rendered_frame = None
                
                for batch_idx, (frame_ids, frames_batch) in enumerate(loader):
                    try:
                        censored_frames = self._process_batch(
                            frames_batch,
                            frame_ids,
                            prev_rendered_frame,
                            result
                        )
                        
                        # Write frames
                        for frame in censored_frames:
                            if frame is not None:
                                writer.write(frame)
                        
                        result.frames_processed += len(frame_ids)
                        
                        if len(censored_frames) > 0:
                            prev_rendered_frame = censored_frames[-1]
                        
                        if progress_callback:
                            progress_callback(result.frames_processed, result.total_frames)
                        
                        # Progress logging
                        if batch_idx % 10 == 0:
                            pct = (result.frames_processed / result.total_frames * 100) if result.total_frames > 0 else 0
                            logger.info(f"Progress: {result.frames_processed}/{result.total_frames} ({pct:.1f}%)")
                    
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} error: {e}", exc_info=True)
                        result.errors.append(f"Batch {batch_idx}: {str(e)}")
                        continue
            
            # Get loader stats
            loader_stats = loader.get_stats()
            result.frames_skipped = loader_stats['frames_skipped_quality'] + loader_stats['frames_skipped_stride']
    
    def _process_batch(
        self,
        frames_batch: np.ndarray,
        frame_ids: List[int],
        prev_rendered_frame: Optional[np.ndarray],
        result: ProcessingResult
    ) -> List[np.ndarray]:
        """Process batch through detection -> tracking -> rendering."""
        
        batch_size = len(frame_ids)
        
        # Detection
        with self.metrics.measure('detection'):
            try:
                nudity_detections = self.nudity_detector.detect(frames_batch)
                face_detections = self.face_detector.detect(frames_batch)
                
                fused_detections = []
                for i in range(batch_size):
                    fused = self.fuser.fuse(
                        nudity_detections[i],
                        face_detections[i]
                    )
                    fused_detections.append(fused)
                    
                    # Update counts
                    for det in fused:
                        self.metrics.record_detection(det)
                        
                        if det.model_name == 'rtdetr':
                            result.nudity_detections += 1
                        elif det.model_name == 'retinaface':
                            result.face_detections += 1
                        
                        result.total_detections += 1
            
            except Exception as e:
                logger.error(f"Detection failed: {e}", exc_info=True)
                return [frames_batch[i] for i in range(batch_size)]
        
        # Tracking
        with self.metrics.measure('tracking'):
            try:
                all_tracks = []
                for i in range(batch_size):
                    tracks = self.tracker.update(
                        fused_detections[i],
                        embeddings=None,
                        frame_idx=frame_ids[i]
                    )
                    all_tracks.append(tracks)
                    
                    for track in tracks:
                        self.metrics.record_track(track)
            
            except Exception as e:
                logger.error(f"Tracking failed: {e}", exc_info=True)
                all_tracks = [fused_detections[i] for i in range(batch_size)]
        
            # Rendering
        with self.metrics.measure('rendering'):
            try:
                censored_frames = []
                prev_frame = prev_rendered_frame
                
                for i in range(batch_size):
                    frame = frames_batch[i]
                    tracks = all_tracks[i]
                    raw_detections = fused_detections[i]
                    
                    # ============================================================
                    # KEY FIX: Combine raw detections + active tracks
                    # ============================================================
                    
                    # Get active tracked detections
                    track_detections = self._tracks_to_detections(tracks)
                    
                    # Find raw detections that are NOT being tracked
                    untracked_detections = self._find_untracked_detections(
                        raw_detections,
                        track_detections
                    )
                    
                    # COMBINE: Tracked objects + Untracked detections
                    all_detections_to_render = track_detections + untracked_detections
                    
                    logger.debug(
                        f"Frame {frame_ids[i]}: "
                        f"{len(track_detections)} tracked + "
                        f"{len(untracked_detections)} untracked = "
                        f"{len(all_detections_to_render)} total"
                    )
                    
                    # ============================================================
                    
                    # Render with combined detections
                    censored = self.renderer.render(
                        frame,
                        all_detections_to_render,  # ← FIXED: Now includes raw detections!
                        prev_frame=prev_frame,
                        merge_by_class=True
                    )
                    
                    if censored is None:
                        logger.error(f"Renderer returned None for frame {frame_ids[i]}")
                        censored = frame.copy()
                    
                    censored_frames.append(censored)
                    prev_frame = censored
                
                return censored_frames
            
            except Exception as e:
                logger.error(f"Rendering failed: {e}", exc_info=True)
                return [frames_batch[i] for i in range(batch_size)]


    def _find_untracked_detections(
        self,
        raw_detections: List[Detection],
        tracked_detections: List[Detection],
        iou_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Find raw detections that are NOT already being tracked.
        
        This catches objects in first 1-2 frames before tracker activates.
        
        Args:
            raw_detections: Direct model output
            tracked_detections: Active track detections
            iou_threshold: IoU threshold for matching
        
        Returns:
            Detections that need immediate rendering (not tracked yet)
        """
        untracked = []
        
        for raw_det in raw_detections:
            is_tracked = False
            
            # Check if this detection is covered by an active track
            for track_det in tracked_detections:
                # Same class?
                if raw_det.class_name != track_det.class_name:
                    continue
                
                # Calculate IoU
                iou = raw_det.iou(track_det)
                
                # If IoU > threshold, it's being tracked
                if iou > iou_threshold:
                    is_tracked = True
                    break
            
            # If not tracked, add to untracked list
            if not is_tracked:
                # Mark as untracked for debugging
                untracked_det = Detection(
                    bbox=raw_det.bbox,
                    confidence=raw_det.confidence,
                    class_id=raw_det.class_id,
                    class_name=raw_det.class_name,
                    frame_id=raw_det.frame_id,
                    model_name=f"{raw_det.model_name}_untracked"  # Mark for debugging
                )
                untracked.append(untracked_det)
        
        return untracked
    
    def _tracks_to_detections(self, tracks: List[Track]) -> List[Detection]:
        """Convert Track objects to Detection objects."""
        detections = []
        
        for track in tracks:
            if not hasattr(track, 'bbox') or track.bbox is None:
                continue
            
            try:
                detection = Detection(
                    bbox=tuple(int(x) for x in track.bbox),
                    confidence=float(getattr(track, 'confidence', 0.5)),
                    class_id=0,
                    class_name=getattr(track, 'class_name', 'unknown'),
                    frame_id=getattr(track, 'last_updated_frame', 0) or 0,
                    model_name='tracker'
                )
                detections.append(detection)
            except Exception as e:
                logger.warning(f"Failed to convert track: {e}")
                continue
        
        return detections
    
    def _export_metadata(self, result: ProcessingResult, metrics_report: dict) -> str:
        """Export processing metadata to JSON."""
        metadata_path = Path(result.output_path).with_suffix('.metadata.json')
        
        metadata = {
            'input_video': result.input_path,
            'output_video': result.output_path,
            'processing_stats': result.to_dict(),
            'performance_metrics': metrics_report,
            'configuration': self.config.to_dict(),
            'renderer_info': self.renderer.get_strategy_info()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata exported: {metadata_path}")
        return str(metadata_path)
    
    def _preserve_audio(self, input_path: Path, output_path: Path):
        """Preserve audio using ffmpeg."""
        try:
            import subprocess
            import time
            
            # Wait for file to finalize
            logger.info("Waiting for video file to finalize...")
            time.sleep(2)
            
            if not output_path.exists():
                logger.warning(f"Output file doesn't exist: {output_path}")
                return
            
            temp_output = output_path.with_suffix('.temp.mp4')
            
            cmd = [
                'ffmpeg',
                '-i', str(output_path),
                '-i', str(input_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                '-y',
                str(temp_output)
            ]
            
            logger.info("Merging audio with ffmpeg...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                text=True
            )
            
            if result.returncode == 0 and temp_output.exists():
                temp_size = temp_output.stat().st_size
                if temp_size > 0:
                    output_path.unlink()
                    temp_output.rename(output_path)
                    logger.info("✓ Audio preserved successfully")
                else:
                    logger.warning("Audio-merged file is empty")
                    temp_output.unlink(missing_ok=True)
            else:
                logger.warning(f"FFmpeg failed: {result.stderr[:500]}")
                temp_output.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Audio preservation error: {e}")
    
    def _log_results(self, result: ProcessingResult):
        """Log processing results."""
        logger.info("\n" + "=" * 70)
        logger.info("✓ PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nOutput: {result.output_path}")
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Frames processed: {result.frames_processed}/{result.total_frames}")
        logger.info(f"  Processing time:  {result.processing_time_seconds:.1f}s")
        logger.info(f"  Average FPS:      {result.fps:.1f}")
        logger.info(f"\n🎯 Detections:")
        logger.info(f"  Total:   {result.total_detections}")
        logger.info(f"  Nudity:  {result.nudity_detections}")
        logger.info(f"  Faces:   {result.face_detections}")
        logger.info(f"  Tracks:  {result.total_tracks}")
        logger.info(f"\n⚡ Performance:")
        logger.info(f"  Detection: {result.avg_detection_time_ms:.1f}ms")
        logger.info(f"  Tracking:  {result.avg_tracking_time_ms:.1f}ms")
        logger.info(f"  Rendering: {result.avg_rendering_time_ms:.1f}ms")
        
        if result.errors:
            logger.warning(f"\n⚠️  Errors: {len(result.errors)}")
        
        logger.info("=" * 70)