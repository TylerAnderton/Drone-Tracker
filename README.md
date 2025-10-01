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

Notes:
- Ensure NVIDIA drivers, CUDA, and TensorRT are installed for `.engine` inference.
- Use Anti-UAV frames for INT8 calibration later if needed.
  ## Live Streaming (Redis Streams)

Minimal 3-stage app on localhost using Redis Streams.

- **Producer**: Webcam → `video:frames`
- **Worker**: YOLO inference → `video:results`
- **Renderer**: display + save MP4

### 1) Install Redis (Ubuntu)
```bash
sudo apt update && sudo apt install -y redis-server
sudo systemctl enable --now redis-server
redis-cli PING  # -> PONG
```

### 2) Run the streaming demo (all roles)
```bash
python -m src.app.run_streaming
```
### 3) Run roles separately (optional)
```bash
# A: Producer
python -m src.app.run_streaming --role producer
# B: Worker
python -m src.app.run_streaming --role worker
# C: Renderer
python -m src.app.run_streaming --role renderer
```
### What you'll see
- **Side-by-side**: input | annotated with boxes and yaw/pitch
- **Latency overlays**: worker pipeline time, and end-to-end (capture→render)
- **Saved video**: `outputs/stream_<timestamp>.mp4` when `--save` is used
