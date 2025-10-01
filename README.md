# Edge-Deployable Drone Detection & Tracking
  
  A minimal, interview-ready pipeline to run real-time object detection & tracking on recorded videos, optimized with TensorRT on NVIDIA GPUs.
  
  - Detector: Ultralytics YOLOv8 (n/s variants recommended)
  - Tracker: ByteTrack (via Ultralytics tracking mode)
  - Optimizer: TensorRT (FP16 by default; INT8 optional)
  - Input: recorded video files (webcam optional later)

## Quickstart

1) Create env and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) (Optional) Export TensorRT engine (FP16)
  
  ```bash
  python -m deploy.trt.export --half
  # -> models/export/yolov8n.engine
  ```

3) Run detection + tracking on a video
  
  ```bash
  # PyTorch (uses config/runtime.yaml defaults)
  python -m src.app.run_demo --source /path/to/video.mp4 --save

  # TensorRT (override model only)
  python -m src.app.run_demo --model models/export/yolov8n.engine --source /path/to/video.mp4 --save
  ```

4) Benchmark latency/FPS
  
  ```bash
  # PyTorch (defaults)
  python -m src.app.metrics --source /path/to/video.mp4 --mode detect
  # TensorRT
  python -m src.app.metrics --model models/export/yolov8n.engine --source /path/to/video.mp4 --mode track
  ```

## Notes on TensorRT

- You must have NVIDIA drivers, CUDA and TensorRT installed for `.engine` inference.
- Ultralytics can export to TensorRT directly. INT8 requires a calibrator; use domain frames (e.g., Anti-UAV) for best results.
## Config

See `config/runtime.yaml` for default thresholds and runtime parameters.

## Structure

```
docs/
  design.md
  calibration.md
data/
  raw/
  calib/
training/
  yolovX/
models/
  weights/
  export/
src/
  detector/
    yolo.py
  tracker/
  calib/
    cli_calibrate.py
  control/
    mapping.py
  app/
    run_demo.py
    run_streaming.py
    metrics.py
    frames_to_video.py
deploy/
  trt/
    export.py
    README.md
config/
  runtime.yaml
  bytetrack.yaml
  calib.yaml
utils/
  profiling.py
  geometry.py
```

Recommended commands using the new entry points:

```bash
# 1) (Optional) Convert Anti-UAV frame folders to MP4 for quick testing
python -m src.app.frames_to_video --input "/data/Anti-UAV-RGBT/<seq>/visible/*.jpg" --fps 25

# 2) Export FP16 TensorRT engine
python -m deploy.trt.export --half

# 3) Run the end-to-end demo (PyTorch or TensorRT)
python -m src.app.run_demo --source outputs/<seq>.mp4 --save
python -m src.app.run_demo --model models/export/yolov8n.engine --source outputs/<seq>.mp4 --save

# 4) Benchmark detect/track latency
python -m src.app.metrics --source outputs/<seq>.mp4 --mode detect
python -m src.app.metrics --model models/export/yolov8n.engine --source outputs/<seq>.mp4 --mode track
```

Notes:
- Ensure NVIDIA drivers, CUDA, and TensorRT are installed for `.engine` inference.
- Use Anti-UAV frames for INT8 calibration later if needed.
  ## Live Streaming (Redis Streams)
  
  Turn the pipeline into a 3-stage streaming app on localhost using Redis Streams.
  
  - **Producer**: Webcam (OpenCV) → publishes JPEG frames to `video:frames`
  - **Worker**: consumes frames → YOLOv8 ByteTrack → publishes annotated JPEG (side-by-side) to `video:results`
  - **Renderer**: consumes results → displays with latency overlays and optionally saves MP4
  ### 1) Install and start Redis (Ubuntu)
  ```bash
  sudo apt update && sudo apt install -y redis-server
  sudo systemctl enable --now redis-server
  redis-cli PING  # -> PONG
  ```
  ### 2) Run everything in one process (fastest path)
  ```bash
  # CPU (default model yolov8n.pt)
  python -m src.app.run_streaming --role all --save --cam 0
  # TensorRT (GPU)
  python -m src.app.run_streaming --role all --model models/export/yolov8n.engine --device 0 --save --cam 0
  ```
  If your external webcam is not index 0, try `--cam 1` (check with `ls /dev/video*` or `v4l2-ctl --list-devices`).
  ### 3) Or run decoupled roles in separate terminals
  ```bash
  # Terminal A: Producer (webcam -> Redis Frames)
  python -m src.app.run_streaming --role producer --cam 0
  # Terminal B: Worker (Frames -> YOLO track -> Results)
  python -m src.app.run_streaming --role worker --model yolov8n.pt --device cpu
  # or TensorRT
  python -m src.app.run_streaming --role worker --model models/export/yolov8n.engine --device 0
  # Terminal C: Renderer (Results -> display/save)
  python -m src.app.run_streaming --role renderer --save --fps_out 20
  ```
  ### What you'll see
  - **Side-by-side**: input frame | annotated output (boxes, IDs, yaw/pitch per box)
  - **Per-stage latency**: Ultralytics pipeline time on the worker, plus end-to-end (capture→render) on the renderer
  - **Saved video**: MP4 under `outputs/stream_<timestamp>.mp4` when `--save` is set
  ### Options and knobs
  - **Streams**: `--frames_stream video:frames`, `--results_stream video:results` (change if running multiple experiments)
  - **Backpressure**: limit backlog via `--maxlen 2000`; the producer will drop oldest entries (approximate MAXLEN) automatically
  - **Compression**: webcam frames encoded as JPEG (`--jpeg_quality 80`); adjust for speed/quality trade-off
  - **Display**: fit-to-window via `--display_max_w`/`--display_max_h`; renderer FPS target via `--fps_out`
  - **Calibration**: yaw/pitch overlay uses `config/calib.yaml`; overlay silently skips if calibration is missing/invalid
  ### Observability
  Check queue depth (backlog) in Redis:
  ```bash
  redis-cli XLEN video:frames
  redis-cli XLEN video:results
  ```
  ### Mapping to Kafka (later)
  - The `producer`/`worker`/`renderer` boundaries and message contracts match Kafka topics.
  - Swap Redis XADD/XREADGROUP with Kafka produce/consume using the same fields (frame_id, ts_ns, jpeg).
  - Keep at-least-once semantics by ACK/commit after successful processing.
