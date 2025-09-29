import argparse
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark YOLOv8 detection/tracking (PyTorch or TensorRT)")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt or TensorRT .engine model")
    p.add_argument("--source", type=str, required=True, help="Video path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--mode", type=str, choices=["detect", "track"], default="detect")
    p.add_argument("--tracker", type=str, default="config/bytetrack.yaml")
    return p.parse_args()


def summarize_times(name: str, times_ms):
    arr = np.array(times_ms, dtype=np.float64)
    if arr.size == 0:
        print(f"[{name}] No frames measured.")
        return
    print(
        f"[{name}] frames={arr.size} mean={arr.mean():.2f} ms, p50={np.percentile(arr,50):.2f} ms, p95={np.percentile(arr,95):.2f} ms"
    )


def bench_detect(model: YOLO, source: str, imgsz: int, conf: float, device: str):
    times_ms = []
    frames = 0
    for r in model.predict(source=source, imgsz=imgsz, conf=conf, device=device, stream=True, verbose=False):
        times_ms.append(r.speed.get("inference", 0.0))
        frames += 1
    summarize_times("detect.inference", times_ms)


def bench_track(model: YOLO, source: str, imgsz: int, conf: float, device: str, tracker: str):
    times_ms = []
    t_prev = time.perf_counter()
    for _ in model.track(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        tracker=tracker,
        stream=True,
        verbose=False,
        persist=True,
        save=False,
    ):
        t_now = time.perf_counter()
        times_ms.append((t_now - t_prev) * 1000.0)
        t_prev = t_now
    summarize_times("track.e2e_frame", times_ms)


def main():
    args = parse_args()
    model = YOLO(args.model)
    source = Path(args.source).as_posix()

    if args.mode == "detect":
        bench_detect(model, source, args.imgsz, args.conf, args.device)
    else:
        t0 = time.perf_counter()
        bench_track(model, source, args.imgsz, args.conf, args.device, args.tracker)
        dt = time.perf_counter() - t0
        print(f"[track] total_time={dt:.2f}s")


if __name__ == "__main__":
    main()
