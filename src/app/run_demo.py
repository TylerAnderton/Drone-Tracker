import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import yaml
import time
import torch
from src.detector.yolo import YoloDetector
from src.control.mapping import detection_center_to_angles


def parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end: decode → detect → track → visualize/save")
    p.add_argument("--config", type=str, default="config/runtime.yaml", help="Runtime YAML config")
    p.add_argument("--model", type=str, default=None, help="Path to .pt or TensorRT .engine model")
    p.add_argument("--source", type=str, required=True, help="Video file path")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tracker", type=str, default=None, help="Tracker config YAML")
    p.add_argument("--calib", type=str, default=None, help="Calibration YAML for yaw/pitch mapping")
    p.add_argument("--save", action="store_true", help="Save annotated MP4 under outputs/")
    p.add_argument("--fps_out", type=float, default=None, help="Force output FPS (0=use source FPS or fallback)")
    p.add_argument("--display_max_w", type=int, default=1600, help="Max display window width (no effect on saved video)")
    p.add_argument("--display_max_h", type=int, default=900, help="Max display window height (no effect on saved video)")
    return p.parse_args()


def main():
    args = parse_args()
    # Load runtime config and merge with CLI (CLI overrides when provided)
    cfg = {}
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    def pick(key, default=None):
        val = getattr(args, key, None)
        if val is None:
            return cfg.get(key, default)
        return val

    model_path = pick("model", "yolov8n.pt")
    imgsz = int(pick("imgsz", 640))
    conf = float(pick("conf", 0.25))
    iou = float(pick("iou", 0.45))
    device = str(pick("device", "0"))
    tracker_cfg = pick("tracker", "config/bytetrack.yaml")
    calib_yaml = pick("calib", "config/calib.yaml")
    fps = int(pick("fps_out", 20))

    det = YoloDetector(model_path, device=device)
    det.warmup(args.source, imgsz=imgsz, conf=conf)

    # Backend and GPU verification prints
    backend = "tensorrt" if str(model_path).lower().endswith(".engine") else "pytorch"
    print(f"[device] backend={backend}, device={device}")
    if backend == "pytorch":
        try:
            cuda_ok = torch.cuda.is_available()
            print(f"[device] torch.cuda.is_available()={cuda_ok}")
            if cuda_ok and device != "cpu":
                try:
                    dev_index = int(device) if str(device).isdigit() else torch.cuda.current_device()
                except Exception:
                    dev_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev_index)
                total_gb = props.total_memory / (1024**3)
                print(f"[device] torch device={dev_index} name={props.name} vram_total={total_gb:.1f} GB")
        except Exception:
            pass

    # Prepare output writer if saving
    writer = None
    out_path = None
    if args.save:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(args.source)
        base_dir = src_path.parent.name  # immediate directory name, e.g. sequence id
        img_type = src_path.name.split(".")[0] # visible or infrared
        out_path = out_dir / f"{base_dir}_{img_type}_annot.mp4"

        # # Determine output FPS
        # if fps_out_cfg and fps_out_cfg > 0:
        #     fps = float(fps_out_cfg)
        # else:
        #     try:
        #         cap = cv2.VideoCapture(args.source)
        #         cap_fps = cap.get(cv2.CAP_PROP_FPS)
        #         if cap_fps and cap_fps > 0:
        #             fps = cap_fps
        #         cap.release()
        #     except Exception:
        #         pass

    frames = 0
    # Real-time playback target and latency tracking
    target_dt = 1.0 / max(1, fps)
    t_prev = time.perf_counter()
    next_frame_time = t_prev + target_dt
    for r in det.track(
        source=Path(args.source).as_posix(),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        tracker_cfg=tracker_cfg,
        save=False, # Should this match config? # TODO: Investigate save parameters https://docs.ultralytics.com/modes/predict/#inference-arguments
        persist=True,
        # stream=True, # Returns generator of results # Unexpected argument for .track()??
    ):
        # r.orig_img: original BGR frame
        frame = r.orig_img
        if frame is None:
            continue

        # End-to-end per-frame latency (time between results)
        t_now = time.perf_counter()
        lat_ms = (t_now - t_prev) * 1000.0
        t_prev = t_now

        # Pipeline latency (decode + preprocess + inference + NMS + tracking) from Ultralytics profiling
        # Prefer internal per-frame timings when available; fallback to inter-result time
        pipe_ms = None
        try:
            sp = getattr(r, 'speed', None)
            if isinstance(sp, dict) and sp:
                # Sum all measured stages (values are in milliseconds)
                pipe_ms = sum(float(v) for v in sp.values() if v is not None)
        except Exception:
            pipe_ms = None
        if pipe_ms is None:
            pipe_ms = lat_ms

        # Initialize writer lazily with frame size
        # if args.save and writer is None:
        #     h, w = frame.shape[:2]
        #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #     writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        # Start with default YOLO plotted overlays (boxes, IDs)
        vis = r.plot()  # returns BGR image with drawings

        # Overlay yaw/pitch for each tracked box center
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            ids = None
            if hasattr(boxes, 'id') and boxes.id is not None:
                ids = boxes.id.cpu().numpy() if hasattr(boxes.id, 'cpu') else np.array(boxes.id)
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box.astype(float)
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                try:
                    yaw, pitch = detection_center_to_angles(u, v, calib_yaml)
                    yaw_deg = np.degrees(yaw)
                    pitch_deg = np.degrees(pitch)
                    label = f"yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°"
                    if ids is not None and i < len(ids) and ids[i] is not None:
                        label = f"ID {int(ids[i])} | " + label
                    org = (int(x1), max(0, int(y1) - 10))
                    cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception:
                    # If calibration missing/invalid, skip overlay for this box
                    pass

        # Overlay latency (bottom-right with padded background)
        # txt = f"{lat_ms:.1f} ms"
        txt = f"{pipe_ms:.1f} ms"
        (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        m = 10
        x = int(vis.shape[1] - tw - m)
        y = int(vis.shape[0] - m)
        cv2.rectangle(vis, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
        cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Build side-by-side (INPUT | OUTPUT) frame for saving and display
        side = np.hstack([frame, vis])

        # Initialize writer lazily with side-by-side frame size
        if args.save and writer is None:
            sh, sw = side.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (sw, sh))

        if args.save and writer is not None:
            # writer.write(vis)
            writer.write(side)

        # Show side-by-side (input | output) in real-time
        try:
            # Fit display into user-defined bounds without altering saved output quality
            sh, sw = side.shape[:2]
            scale = min(args.display_max_w / max(1, sw), args.display_max_h / max(1, sh), 1.0)
            disp = side if scale >= 0.999 else cv2.resize(side, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Drone tracking live", disp)
            # Throttle to target FPS
            now2 = time.perf_counter()
            delay_ms = int(max(0.0, (next_frame_time - now2)) * 1000)
            key = cv2.waitKey(max(1, delay_ms)) & 0xFF
            next_frame_time += target_dt
            if key in (27, ord('q')):
                break
        except Exception:
            # In headless environments, imshow may fail; continue without display
            pass

        frames += 1
        if frames % 100 == 0:
            print(f"[run_demo] processed frames={frames}")

    if writer is not None:
        writer.release()
        print(f"[run_demo] Saved annotated video: {out_path}")

    cv2.destroyAllWindows()
    print(f"[run_demo] Completed {frames} frames (model={model_path}, imgsz={imgsz})")


if __name__ == "__main__":
    main()
