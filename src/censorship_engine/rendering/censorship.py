"""
Censorship renderer: merges detections, picks strategies, applies censorship,
and supports temporal blending.


- Per-class strategy assignment with fallback
- Robust error handling and validation
- Efficient box merging with configurable IoU
- Strategy instance caching with proper options handling
- Comprehensive logging for debugging
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import logging
from censorship_engine.core.datatypes import Detection
from censorship_engine.rendering.strategies import (
    RenderStrategy,
    get_strategy,
    STRATEGY_REGISTRY
)

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


def _calculate_iou(box1: BBox, box2: BBox) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    
    # Intersection
    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    area_a = max(0, (x2a - x1a) * (y2a - y1a))
    area_b = max(0, (x2b - x1b) * (y2b - y1b))
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0.0


def _merge_overlapping_boxes(boxes: List[BBox], iou_threshold: float = 0.5) -> List[BBox]:
    """
    Greedy merging of overlapping boxes into a single bbox using IoU.
    Iteratively merges boxes until no overlapping pairs remain.
    
    Args:
        boxes: List of bounding boxes (x1, y1, x2, y2)
        iou_threshold: IoU threshold for merging (0.0 to 1.0)
    
    Returns:
        List of merged bounding boxes
    """
    if len(boxes) == 0:
        return []
    
    # Ensure boxes are tuples of ints
    boxes = [tuple(int(x) for x in b) for b in boxes]
    
    merged = True
    iterations = 0
    max_iterations = len(boxes) * 2  # Prevent infinite loops
    
    while merged and iterations < max_iterations:
        merged = False
        iterations += 1
        new_boxes = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            current_box = boxes[i]
            used[i] = True
            
            # Try to merge with remaining boxes
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                iou = _calculate_iou(current_box, boxes[j])
                
                if iou >= iou_threshold:
                    # Merge boxes by taking min/max coordinates
                    x1 = min(current_box[0], boxes[j][0])
                    y1 = min(current_box[1], boxes[j][1])
                    x2 = max(current_box[2], boxes[j][2])
                    y2 = max(current_box[3], boxes[j][3])
                    current_box = (x1, y1, x2, y2)
                    used[j] = True
                    merged = True
            
            new_boxes.append(current_box)
        
        boxes = new_boxes
    
    if iterations >= max_iterations:
        logger.warning(f"Box merging hit max iterations ({max_iterations})")
    
    return boxes


class CensorshipRenderer:
    """
    Production-ready censorship renderer with per-class strategies.
    
    Features:
    - Per-class strategy assignment (e.g., blur for faces, blackout for nudity)
    - Fallback to default strategy for unknown classes
    - Box merging to handle overlapping detections
    - Temporal blending for smooth transitions
    - Comprehensive error handling and logging
    
    Usage:
        renderer = CensorshipRenderer(
            default_strategy='blur',
            strategy_map={'face': 'pixelate', 'breast': 'blackout'},
            strategy_options={
                'blur': {'kernel_size': 51, 'intensity': 1.0},
                'blackout': {'color': (0, 0, 255), 'opacity': 1.0}
            }
        )
        censored = renderer.render(frame, detections)
    """

    def __init__(
        self,
        default_strategy: str = "blur",
        strategy_map: Optional[Dict[str, str]] = None,
        strategy_options: Optional[Dict[str, dict]] = None,
        merge_iou: float = 0.5,
        blend_alpha: float = 0.0,
        emoji_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the censorship renderer.
        
        Args:
            default_strategy: Fallback strategy name for classes not in strategy_map
            strategy_map: Mapping of class_name -> strategy_name
                         Example: {'face': 'blur', 'breast': 'blackout', 'genitals': 'pixelate'}
            strategy_options: Mapping of strategy_name -> kwargs
                            Example: {'blur': {'kernel_size': 51, 'intensity': 1.0}}
            merge_iou: IoU threshold for merging overlapping boxes (0.0 to 1.0)
            blend_alpha: Temporal blending factor (0.0 = no blending, 1.0 = full blend)
            emoji_map: Mapping of class_name -> emoji_png_path for emoji strategy
        """
        self.default_strategy = self._validate_strategy_name(default_strategy)
        self.strategy_map = strategy_map or {}
        self.strategy_options = strategy_options or {}
        self.merge_iou = max(0.0, min(1.0, float(merge_iou)))
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))
        self.emoji_map = emoji_map or {}
        
        # Validate strategy map
        self._validate_strategy_map()
        
        # Strategy instance cache: (strategy_name, options_hash) -> strategy_instance
        # We cache based on strategy name AND options to avoid conflicts
        self._strategy_cache: Dict[Tuple[str, str], RenderStrategy] = {}
        
        logger.info(f"CensorshipRenderer initialized:")
        logger.info(f"  Default strategy: {self.default_strategy}")
        logger.info(f"  Strategy map: {self.strategy_map}")
        logger.info(f"  Merge IoU: {self.merge_iou}")
        logger.info(f"  Temporal blend alpha: {self.blend_alpha}")
    
    @staticmethod
    def _validate_strategy_name(name: str) -> str:
        """Validate and normalize strategy name."""
        name = name.lower().strip()
        if name not in STRATEGY_REGISTRY:
            logger.warning(f"Unknown strategy '{name}', defaulting to 'blur'")
            return 'blur'
        return name
    
    def _validate_strategy_map(self):
        """Validate all strategies in strategy_map."""
        for class_name, strategy_name in list(self.strategy_map.items()):
            validated = self._validate_strategy_name(strategy_name)
            if validated != strategy_name:
                logger.warning(f"Strategy '{strategy_name}' for class '{class_name}' "
                             f"validated to '{validated}'")
                self.strategy_map[class_name] = validated
    
    def _get_strategy(self, name: str, class_name: Optional[str] = None) -> RenderStrategy:
        """
        Get or create a strategy instance (with caching).
        
        Args:
            name: Strategy name
            class_name: Optional class name for emoji path lookup
        
        Returns:
            RenderStrategy instance
        """
        name = self._validate_strategy_name(name)
        
        # Get base options for this strategy
        opts = self.strategy_options.get(name, {}).copy() if self.strategy_options else {}
        
        # For emoji strategy, check if class has specific emoji path
        if name == "emoji" and class_name and class_name in self.emoji_map:
            opts['png_path'] = self.emoji_map[class_name]
        elif name == "emoji" and 'png_path' not in opts and len(self.emoji_map) > 0:
            # Fallback to first emoji in map
            opts['png_path'] = next(iter(self.emoji_map.values()))
        
        # Create cache key from options
        options_hash = str(sorted(opts.items()))
        cache_key = (name, options_hash)
        
        # Return cached instance if available
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        # Create new instance
        try:
            strategy = get_strategy(name, **opts)
            
            # Cache it (limit cache size)
            if len(self._strategy_cache) < 100:
                self._strategy_cache[cache_key] = strategy
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating strategy '{name}': {e}")
            # Fallback to blur with no options
            return get_strategy('blur')
    
    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        prev_frame: Optional[np.ndarray] = None,
        merge_by_class: bool = True
    ) -> np.ndarray:
        """
        Apply censorship on frame using detections.
        
        Args:
            frame: (H,W,3) BGR image numpy array
            detections: List of Detection objects
            prev_frame: Optional previous rendered frame for temporal blending
            merge_by_class: Whether to merge boxes per class (True) or globally (False)
        
        Returns:
            Censored frame (BGR numpy array)
        
        Raises:
            ValueError: If frame is invalid
        """
        # Validate frame
        if frame is None:
            raise ValueError("Frame is None")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be numpy.ndarray, got {type(frame)}")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be (H,W,3), got shape {frame.shape}")
        
        # Copy frame for modification
        out = frame.copy()
        
        # Early exit if no detections
        if len(detections) == 0:
            if prev_frame is not None and self.blend_alpha > 0:
                return self._temporal_blend(prev_frame, out, alpha=self.blend_alpha)
            return out
        
        # Group and merge boxes
        try:
            merged_boxes = self._prepare_boxes(detections, merge_by_class)
        except Exception as e:
            logger.error(f"Error preparing boxes: {e}")
            return out
        
        # Sort boxes by area (largest first) for better visual quality
        merged_boxes.sort(
            key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]),
            reverse=True
        )
        
        # Apply censorship strategies
        for bbox, class_name in merged_boxes:
            try:
                # Determine strategy for this class
                strategy_name = self.strategy_map.get(class_name, self.default_strategy) \
                               if class_name is not None else self.default_strategy
                
                # Get strategy instance
                strategy = self._get_strategy(strategy_name, class_name)
                
                # Apply censorship
                out = strategy.apply(out, bbox)
                
            except Exception as e:
                logger.error(f"Error applying strategy '{strategy_name}' to class '{class_name}': {e}")
                # Continue with next box (graceful degradation)
                continue
        
        # Temporal blending if enabled
        if prev_frame is not None and self.blend_alpha > 0:
            try:
                out = self._temporal_blend(prev_frame, out, alpha=self.blend_alpha)
            except Exception as e:
                logger.error(f"Error in temporal blending: {e}")
        
        return out
    # Add this method to your CensorshipRenderer class in censorship.py

    def get_strategy_info(self) -> Dict[str, any]:
        """
        Get information about configured strategies.
        
        Returns:
            Dictionary with strategy configuration info
        """
        # Get list of available strategies
        available_strategies = ['blur', 'gaussian', 'gauss', 'pixelate', 'pixel', 
                            'blackbox', 'black', 'box', 'blackout', 'emoji']
        
        # Get cached strategies
        cached_strategies = list(self._strategies.keys()) if hasattr(self, '_strategies') else []
        
        return {
            'default_strategy': self.default_strategy,
            'class_strategies': self.strategy_map,
            'available_strategies': available_strategies,
            'cached_strategies': cached_strategies,
            'merge_iou': self.merge_iou,
            'temporal_blend_alpha': self.blend_alpha,
        }

    def _prepare_boxes(
        self,
        detections: List[Detection],
        merge_by_class: bool
    ) -> List[Tuple[BBox, Optional[str]]]:
        """
        Group detections by class and merge overlapping boxes.
        
        Args:
            detections: List of Detection objects
            merge_by_class: Whether to merge per-class or globally
        
        Returns:
            List of (bbox, class_name) tuples
        """
        if merge_by_class:
            # Group by class, merge within each class
            class_groups: Dict[str, List[BBox]] = {}
            
            for d in detections:
                class_name = d.class_name or 'unknown'
                bbox = tuple(int(x) for x in d.bbox)
                class_groups.setdefault(class_name, []).append(bbox)
            
            # Merge within each class
            merged_boxes = []
            for class_name, boxes in class_groups.items():
                merged = _merge_overlapping_boxes(boxes, iou_threshold=self.merge_iou)
                for bbox in merged:
                    merged_boxes.append((bbox, class_name))
        else:
            # Merge all boxes together (lose class info)
            boxes = [tuple(int(x) for x in d.bbox) for d in detections]
            merged = _merge_overlapping_boxes(boxes, iou_threshold=self.merge_iou)
            merged_boxes = [(bbox, None) for bbox in merged]
        
        return merged_boxes
    
    def _temporal_blend(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        alpha: float = 0.2
    ) -> np.ndarray:
        """
        Blend current frame with previous frame for temporal smoothing.
        
        Args:
            prev_frame: Previous rendered frame
            curr_frame: Current rendered frame
            alpha: Blending factor (0.0 = use prev, 1.0 = use curr)
        
        Returns:
            Blended frame
        """
        if prev_frame is None:
            return curr_frame
        
        # Resize prev frame if shape mismatch
        if prev_frame.shape != curr_frame.shape:
            try:
                prev_frame = cv2.resize(
                    prev_frame,
                    (curr_frame.shape[1], curr_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            except Exception as e:
                logger.error(f"Error resizing prev_frame: {e}")
                return curr_frame
        
        # Clamp alpha
        alpha = max(0.0, min(1.0, float(alpha)))
        
        try:
            # Weighted blend
            blended = cv2.addWeighted(
                curr_frame.astype(np.float32), alpha,
                prev_frame.astype(np.float32), 1.0 - alpha,
                0
            ).astype(np.uint8)
            return blended
        except Exception as e:
            logger.error(f"Error blending frames: {e}")
            return curr_frame
    
    def clear_cache(self):
        """Clear strategy instance cache (useful for memory management)."""
        self._strategy_cache.clear()
        logger.info("Strategy cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            'cached_strategies': len(self._strategy_cache),
            'configured_strategies': len(set(self.strategy_map.values())) + 1,  # +1 for default
        }