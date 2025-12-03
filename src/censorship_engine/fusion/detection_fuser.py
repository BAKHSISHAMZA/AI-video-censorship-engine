# src/censorship_engine/fusion/detection_fuser.py

from typing import List
import logging
from censorship_engine.core.datatypes import Detection

logger = logging.getLogger(__name__)


class DetectionFuser:
    def __init__(self, iou_threshold: float = 0.5):
        # TODO: Initialize
        self.iou_threshold = iou_threshold
    
    def fuse(
        self,
        nudity_detections: List[Detection],
        face_detections: List[Detection]
    ) -> List[Detection]:
        
        """
        Fuse detections from nudity detector and face detector.
        
        Args:
            nudity_detections: Detections from RT-DETR
            face_detections: Detections from RetinaFace
        
        Returns:
            Merged and filtered detection list
        
        Example:
            >>> fuser = DetectionFuser(iou_threshold=0.5)
            >>> nudity = [det1, det2, det3]
            >>> faces = [det4, det5]
            >>> merged = fuser.fuse(nudity, faces)
            >>> print(len(merged))
            5  # Or fewer if duplicates were suppressed
        """
        all_detections = nudity_detections + face_detections
        filtered_detections = self._apply_class_wise_nms(all_detections)
        return filtered_detections
    
    def _apply_class_wise_nms(self, detections: List[Detection]) -> List[Detection]:

        """Apply NMS separately for each class."""
        # Group by class
        by_class = {}
        for det in detections:
            if det.class_name not in by_class:
                by_class[det.class_name] = []
            by_class[det.class_name].append(det)
        
        # Apply NMS per class
        filtered = []
        for class_name, class_dets in by_class.items():
            class_filtered = self._apply_nms(class_dets)
            filtered.extend(class_filtered)
        
        # Sort by confidence
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        
        return filtered
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:

        if len(detections) == 0:
            return []
        
        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept = []
        while len(sorted_dets) > 0:
            best = sorted_dets.pop(0)
            kept.append(best)
            
            # Remove overlapping detections
            sorted_dets = [
                det for det in sorted_dets
                if self._iou(best.bbox, det.bbox) < self.iou_threshold
            ]
        
        return kept
    
    def _iou(self, bbox1: tuple, bbox2: tuple) -> float:
        
        x1_a, y1_a, x2_a, y2_a = bbox1
        x1_b, y1_b, x2_b, y2_b = bbox2
        
        # Intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0