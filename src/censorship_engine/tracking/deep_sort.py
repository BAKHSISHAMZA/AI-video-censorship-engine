"""
Lightweight DeepSORT-style tracker for censorship pipeline.

Design goals:
- Works with your Detection dataclass.
- Uses a small Kalman filter per track (constant velocity in x/y + width/height).
- Matches detections to tracks using IoU by default.
- Optionally accepts an embedding extractor function (detection -> vector)
  to use cosine-distance appearance matching combined with IoU.
- Uses Hungarian assignment when scipy is available, else a greedy fallback.
- Minimal dependencies: numpy only (scipy optional).

API:
    tracker = DeepSortTracker(max_age=30, iou_threshold=0.3, use_cosine=False)
    tracks = tracker.update(detections)  # returns list of active Track objects

Each Track object contains:
    - track_id: unique int
    - bbox: last bbox (x1,y1,x2,y2)
    - age: frames since creation
    - time_since_update: frames since last matched
    - hits: number of total matches
    - kalman_state: internal (opaque)
    - embedding: last appearance vector (if provided)

"""

from __future__ import annotations
from typing import List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass, field
import math
import logging
import itertools

logger = logging.getLogger(__name__)

# try to import Hungarian solver
try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    _HUNGARIAN = False
    logger.debug("scipy not available - falling back to greedy assignment")


# -------------------------
# Kalman filter utilities
# -------------------------
def _convert_bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert bbox (x1,y1,x2,y2) to z state [cx, cy, s, r] where
       cx,cy center; s = area; r = aspect ratio (w/h).
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([cx, cy, s, r], dtype=float)


def _convert_x_to_bbox(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert Kalman state vector x to bbox (x1,y1,x2,y2)."""
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = math.sqrt(max(1e-6, s * r))
    h = max(1e-6, s / w)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (x1, y1, x2, y2)


class SimpleKalman:
    """
    Tiny Kalman filter for bbox tracking in the space [cx,cy,s,r].
    State vector: [cx, cy, s, r, vx, vy, vs, vr] (8 dims)
    Only minimal tuning; good enough for smoothing and prediction.
    """

    def __init__(self):
        # State vector (8): position + velocity
        self._x = np.zeros((8, 1), dtype=float)
        # Covariance
        self._P = np.eye(8, dtype=float) * 10.0
        # Motion matrix (constant velocity model)
        dt = 1.0
        self._F = np.eye(8, dtype=float)
        for i in range(4):
            self._F[i, i + 4] = dt
        # Measurement matrix maps state to measurement z = [cx,cy,s,r]
        self._H = np.zeros((4, 8), dtype=float)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0
        # Process noise
        self._Q = np.eye(8, dtype=float) * 1.0
        # Measurement noise
        self._R = np.eye(4, dtype=float) * 10.0

        self.age = 0
        self.time_since_update = 0

    def initiate(self, measurement: np.ndarray):
        """Initialize state with measurement z (4x)."""
        z = measurement.reshape((4, 1))
        self._x[0:4, 0:1] = z
        self._x[4:, 0:1] = 0.0
        self._P = np.eye(8, dtype=float) * 10.0
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        """Predict next state."""
        self._x = np.dot(self._F, self._x)
        self._P = np.dot(np.dot(self._F, self._P), self._F.T) + self._Q
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement: np.ndarray):
        """Correct state with measurement z (4x)."""
        z = measurement.reshape((4, 1))
        S = np.dot(self._H, np.dot(self._P, self._H.T)) + self._R
        K = np.dot(np.dot(self._P, self._H.T), np.linalg.inv(S))
        y = z - np.dot(self._H, self._x)
        self._x = self._x + np.dot(K, y)
        I = np.eye(self._P.shape[0])
        self._P = np.dot(I - np.dot(K, self._H), self._P)
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        return self._x.copy()


# -------------------------
# Track dataclass
# -------------------------
@dataclass
class Track:
    track_id: int
    kalman: SimpleKalman
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    embedding: Optional[np.ndarray] = None
    last_updated_frame: Optional[int] = None

    def to_detection(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "hits": self.hits,
        }


# -------------------------
# Helper: IoU
# -------------------------
def iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# -------------------------
# Matching utilities
# -------------------------
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 1.0
    a = a.flatten()
    b = b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _match_cost_matrix(detections: List[Tuple], tracks: List[Track],
                       use_cosine: bool, lambda_cosine: float,
                       embed_list: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Build a cost matrix between detections and tracks.
    Cost is 1 - IoU (lower better); if embeddings used, combine:
    cost = (1 - iou) * (1 - lambda) + cosine_dist * lambda
    """
    N = len(detections)
    M = len(tracks)
    cost = np.zeros((N, M), dtype=float)
    for i, det in enumerate(detections):
        dbbox = det.bbox if hasattr(det, "bbox") else det[0]
        for j, tr in enumerate(tracks):
            iou_score = iou(dbbox, tr.bbox)
            if use_cosine and (embed_list is not None):
                cosd = _cosine_distance(embed_list[i], tr.embedding)
                combined = (1.0 - iou_score) * (1.0 - lambda_cosine) + cosd * lambda_cosine
                cost[i, j] = combined
            else:
                cost[i, j] = 1.0 - iou_score
    return cost


def _linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return matches, unmatched_detections, unmatched_tracks."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    if _HUNGARIAN:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_trs = list(range(cost_matrix.shape[1]))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                continue
            matches.append((r, c))
            unmatched_dets.remove(r)
            unmatched_trs.remove(c)
        return matches, unmatched_dets, unmatched_trs
    else:
        # greedy fallback: sort pairs by cost asc and pick
        pairs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                pairs.append((cost_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        matched_dets = set()
        matched_trs = set()
        matches = []
        for cst, i, j in pairs:
            if cst > thresh:
                break
            if i in matched_dets or j in matched_trs:
                continue
            matches.append((i, j))
            matched_dets.add(i)
            matched_trs.add(j)
        unmatched_dets = [i for i in range(cost_matrix.shape[0]) if i not in matched_dets]
        unmatched_trs = [j for j in range(cost_matrix.shape[1]) if j not in matched_trs]
        return matches, unmatched_dets, unmatched_trs


# -------------------------
# Tracker
# -------------------------
class DeepSortTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        use_cosine: bool = False,
        lambda_cosine: float = 0.2,
        max_tracks: Optional[int] = None,
    ):
        """
        Args:
            max_age: frames to keep unmatched track alive
            min_hits: number of hits before a track is considered confirmed
            iou_threshold: max allowed cost in matching (lower = stricter)
            use_cosine: whether to use appearance embeddings in matching
            lambda_cosine: weight of appearance (0..1) when combining
            max_tracks: optionally limit number of active tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_cosine = use_cosine
        self.lambda_cosine = lambda_cosine
        self.max_tracks = max_tracks

        self.tracks: List[Track] = []
        self._next_id = 1

    def _predict_tracks(self):
        for t in self.tracks:
            t.kalman.predict()
            t.bbox = _convert_x_to_bbox(t.kalman.get_state().flatten())
            t.age += 1
            t.time_since_update += 1

    def _init_track(self, detection, embedding: Optional[np.ndarray], frame_idx: Optional[int]):
        kf = SimpleKalman()
        z = _convert_bbox_to_z(detection.bbox)
        kf.initiate(z)
        track = Track(
            track_id=self._next_id,
            kalman=kf,
            bbox=detection.bbox,
            class_name=detection.class_name,
            confidence=detection.confidence,
            hits=1,
            age=0,
            time_since_update=0,
            embedding=embedding,
            last_updated_frame=frame_idx
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def update(self, detections: List, embeddings: Optional[List[np.ndarray]] = None, frame_idx: Optional[int] = None) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: list of Detection objects (expects .bbox, .confidence, .class_name)
            embeddings: optional list of np arrays corresponding to detections
            frame_idx: optional frame index (for debugging/timestamps)

        Returns:
            list of active confirmed Track objects
        """
        # Predict existing tracks
        self._predict_tracks()

        # If no tracks and no detections: nothing to do
        if len(self.tracks) == 0 and len(detections) == 0:
            return []

        # Build cost matrix between detections and current tracks
        cost_matrix = _match_cost_matrix(detections, self.tracks, self.use_cosine, self.lambda_cosine, embeddings)

        # Perform assignment
        matches, unmatched_dets, unmatched_trs = _linear_assignment(cost_matrix, thresh=self.iou_threshold)

        # Update matched tracks
        for det_idx, tr_idx in matches:
            det = detections[det_idx]
            tr = self.tracks[tr_idx]
            z = _convert_bbox_to_z(det.bbox)
            tr.kalman.update(z)
            tr.bbox = det.bbox
            tr.confidence = det.confidence
            tr.hits += 1
            tr.time_since_update = 0
            tr.last_updated_frame = frame_idx
            # optionally update embedding
            if self.use_cosine and embeddings is not None:
                tr.embedding = embeddings[det_idx]

        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            det = detections[d]
            emb = embeddings[d] if (embeddings is not None and len(embeddings) > d) else None
            self._init_track(det, emb, frame_idx)

        # Prune dead tracks and build output list
        remaining_tracks = []
        for tr in self.tracks:
            if tr.time_since_update <= self.max_age and (tr.hits >= self.min_hits or tr.time_since_update == 0):
                remaining_tracks.append(tr)
        # remove old tracks
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        # Optionally limit track count
        if self.max_tracks is not None and len(remaining_tracks) > self.max_tracks:
            # keep highest hit counts
            remaining_tracks.sort(key=lambda t: (t.hits, t.confidence), reverse=True)
            remaining_tracks = remaining_tracks[: self.max_tracks]

        return remaining_tracks

    def reset(self):
        self.tracks = []
        self._next_id = 1

