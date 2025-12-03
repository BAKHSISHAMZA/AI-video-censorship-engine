import pytest
from dataclasses import dataclass
from censorship_engine.tracking.deep_sort import DeepSortTracker
import numpy as np

# Mock Detection identical to your real one
@dataclass
class Detection:
    bbox: tuple   # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    model_name: str = "test-model"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def make_det(x, y, size=50, frame_id=0):
    """Create a square bbox centered at (x, y)."""
    half = size / 2
    return Detection(
        bbox=(x - half, y - half, x + half, y + half),
        confidence=0.9,
        class_id=0,
        class_name="test",
        frame_id=frame_id,
    )


# --------------------------------------------------
# TESTS
# --------------------------------------------------

def test_single_track_creation():
    tracker = DeepSortTracker(max_age=5, iou_threshold=0.3)

    det = make_det(100, 100)
    tracks = tracker.update([det])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 1
    assert tracks[0].time_since_update == 0


def test_track_persistence_across_frames():
    tracker = DeepSortTracker(max_age=5, iou_threshold=0.3)

    # Frame 1
    tracks1 = tracker.update([make_det(100, 100)])
    tid = tracks1[0].track_id

    # Frame 2 - small movement (still IoU match)
    tracks2 = tracker.update([make_det(105, 105)])

    assert len(tracks2) == 1
    assert tracks2[0].track_id == tid, "Track ID should persist across frames"
    assert tracks2[0].hits == 2
    assert tracks2[0].time_since_update == 0


def test_track_deletion_after_max_age():
    tracker = DeepSortTracker(max_age=2)

    # Create a track
    tracker.update([make_det(100, 100)])

    # 3 empty frames → should delete (max_age=2)
    tracker.update([])
    tracker.update([])
    remaining = tracker.update([])

    assert len(remaining) == 0, "Track should be removed after max_age frames"


def test_two_tracks_independent():
    tracker = DeepSortTracker(max_age=5)

    det1 = make_det(100, 100)
    det2 = make_det(300, 300)

    tracks = tracker.update([det1, det2])

    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert ids == {1, 2}

    # Move both slightly
    tracks2 = tracker.update([
        make_det(105, 105),
        make_det(295, 295)
    ])

    ids2 = {t.track_id for t in tracks2}
    assert ids2 == ids, "Both IDs must stay consistent"


def test_iou_matching_vs_creation():
    tracker = DeepSortTracker(max_age=5, iou_threshold=0.3)

    # Frame 1: create one track
    tracker.update([make_det(100, 100)])

    # Frame 2: large movement → IoU too low → new track
    tracks = tracker.update([make_det(300, 300)])

    ids = [t.track_id for t in tracks]
    assert len(ids) == 2
    assert set(ids) == {1, 2}, "New track must be created when IoU is low"
