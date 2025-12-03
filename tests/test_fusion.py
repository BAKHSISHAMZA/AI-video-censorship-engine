import pytest
from censorship_engine.fusion.detection_fuser import DetectionFuser
from censorship_engine.core.datatypes import Detection


def make_det(x1, y1, x2, y2, conf, cls_name, frame=0, model="test"):
    return Detection(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        class_id=0 if cls_name == "nudity" else 1,
        class_name=cls_name,
        frame_id=frame,
        model_name=model
    )


def test_fusion_no_overlap():
    fuser = DetectionFuser(iou_threshold=0.5)

    nudity = [
        make_det(10, 10, 50, 50, 0.9, "nudity"),
    ]
    faces = [
        make_det(100, 100, 150, 150, 0.95, "face"),
    ]

    fused = fuser.fuse(nudity, faces)

    assert len(fused) == 2
    assert any(det.class_name == "nudity" for det in fused)
    assert any(det.class_name == "face" for det in fused)


def test_fusion_with_overlap_same_class():
    fuser = DetectionFuser(iou_threshold=0.5)

    # Two overlapping nudity detections (same class)
    det1 = make_det(10, 10, 60, 60, 0.95, "nudity")
    det2 = make_det(15, 15, 55, 55, 0.6, "nudity")  # fully inside det1

    fused = fuser.fuse([det1, det2], [])

    # Only highest confidence kept
    assert len(fused) == 1
    assert fused[0].confidence == 0.95


def test_fusion_classwise_nms():
    fuser = DetectionFuser(iou_threshold=0.5)

    nudity_dets = [
        make_det(0, 0, 100, 100, 0.9, "nudity"),
        make_det(10, 10, 90, 90, 0.8, "nudity"),
    ]

    face_dets = [
        make_det(200, 200, 260, 260, 0.7, "face"),
        make_det(205, 205, 255, 255, 0.95, "face"),
    ]

    fused = fuser.fuse(nudity_dets, face_dets)

    # NMS should keep:
    # - top nudity (0.9)
    # - top face (0.95)
    assert len(fused) == 2

    confs = [d.confidence for d in fused]
    assert 0.9 in confs
    assert 0.95 in confs
