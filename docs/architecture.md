               ┌────────────────────────┐
               │     Input Video         │
               └────────────┬───────────┘
                            ▼
                ┌────────────────────────┐
                │  Frame Extraction       │
                └────────────┬───────────┘
                            ▼
     ┌─────────────────────────────────────────────────────────────┐
     │                        Detection Stage                       │
     │                                                             │
     │   ┌──────────────┐   ┌──────────────────┐   ┌───────────┐  │
     │   │ RT-DETR       │   │ RetinaFace       │   │ Classifier │  │
     │   │ (nudity)      │   │ (face detect)    │   │ (fallback │  │
     │   └───────┬──────┘   └─────────┬────────┘   └────┬──────┘  │
     └───────────┼────────────────────┼──────────────────┼──────────┘
                 ▼                    ▼                  ▼
        ┌────────────────────────────────────────────────────┐
        │            Detection Fusion & Rules Engine          │
        │ - Combine detections from multiple models           │
        │ - Resolve conflicts & missing detections            │
        │ - Retrospective smoothing (look-back buffer)        │
        └──────────────────────┬──────────────────────────────┘
                               ▼
                ┌─────────────────────────┐
                │    Object Tracking      │
                │ (DeepSORT / ByteTrack)  │
                └─────────────┬──────────┘
                               ▼
                ┌─────────────────────────┐
                │   Censoring Engine      │
                │ - Pixelate / Blur       │
                │ - Mosaic / Blackout     │
                │ - Face anonymization    │
                └─────────────┬──────────┘
                               ▼
                ┌─────────────────────────┐
                │   Frame Reassembly      │
                └─────────────┬──────────┘
                               ▼
                 ┌────────────────────────┐
                 │      Output Video      │
                 └────────────────────────┘
