import pytest
from censorship_engine.core.datatypes import Detection


def test_detection_valid_instantiation():
    det = Detection(
        bbox=(10, 20, 50, 60),
        confidence=0.85,
        class_id=0,
        class_name="nudity",
        frame_id=0,
        model_name="rtdetr"
    )

    assert det.bbox == (10, 20, 50, 60)
    assert det.area == (50 - 10) * (60 - 20)
    assert det.center == ((10 + 50) / 2, (20 + 60) / 2)
    assert det.class_id == 0
    assert det.class_name == "nudity"
    assert 0 <= det.timestamp <= 9999999999


def test_detection_invalid_confidence():
    with pytest.raises(ValueError):
        Detection(
            bbox=(0, 0, 10, 10),
            confidence=1.5,   # invalid
            class_id=1,
            class_name="face",
            frame_id=0,
            model_name="retinaface"
        )


def test_detection_invalid_bbox_length():
    with pytest.raises(ValueError):
        Detection(
            bbox=(0, 0, 10),   # only 3 values
            confidence=0.5,
            class_id=1,
            class_name="face",
            frame_id=0,
            model_name="retinaface"
        )


def test_iou_computation():
    det1 = Detection(
        bbox=(10, 10, 50, 50),
        confidence=0.8,
        class_id=0,
        class_name="nudity",
        frame_id=0,
        model_name="rtdetr"
    )

    det2 = Detection(
        bbox=(30, 30, 70, 70),
        confidence=0.7,
        class_id=0,
        class_name="nudity",
        frame_id=0,
        model_name="rtdetr"
    )

    iou = det1.iou(det2)

    # Expected intersection = 20x20 = 400
    # Area det1 = 1600, det2 = 1600
    # Union = 1600 + 1600 - 400 = 2800
    # IoU ≈ 0.1428
    assert pytest.approx(iou, 0.01) == 400 / 2800
