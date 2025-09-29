import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import yaml
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
    for r in det.track(
        source=Path(args.source).as_posix(),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        tracker_cfg=tracker_cfg,
        save=False,
        persist=True,
    ):
        # r.orig_img: original BGR frame
        frame = r.orig_img
        if frame is None:
            continue

        # Initialize writer lazily with frame size
        if args.save and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

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

        if args.save and writer is not None:
            writer.write(vis)

        frames += 1
        if frames % 100 == 0:
            print(f"[run_demo] processed frames={frames}")

    if writer is not None:
        writer.release()
        print(f"[run_demo] Saved annotated video: {out_path}")

    print(f"[run_demo] Completed {frames} frames (model={args.model}, imgsz={args.imgsz})")


if __name__ == "__main__":
    main()
