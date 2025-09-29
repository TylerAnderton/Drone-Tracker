# System Design

- **Objective**: Low-latency object detection + multi-object tracking suitable for turret control.
- **Pipeline**: decode → preprocess → YOLO detect → NMS → ByteTrack → control mapping → visualize/log.
- **Latency goals**: ≥30–60 FPS @ 640 on RTX 4070 SUPER.
- **Optimizations**: fixed-size inputs, batch=1, TensorRT FP16 (INT8 optional), tracker threshold tuning.
- **Metrics**:
  - Runtime: FPS, per-stage latency (pre/infer/post/track).
  - Quality: precision/recall on small curated subset, IDF1/MOTA (later), track fragmentation.
- **Extensibility**: swap detectors (YOLOv8/10), trackers (ByteTrack/OC-SORT), deploy to Jetson.
