import argparse
import base64
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import redis
import yaml

from src.detector.yolo import YoloDetector
from src.control.mapping import detection_center_to_angles


# ---------------------------
# Helpers
# ---------------------------

def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return img


def ensure_group(r: redis.Redis, stream: str, group: str, start_id: bytes = b"$") -> None:
    try:
        # start_id '$' means start from last entry (ignore history). '0' means from beginning.
        r.xgroup_create(name=stream, groupname=group, id=start_id.decode() if isinstance(start_id, (bytes, bytearray)) else start_id, mkstream=True)
    except redis.ResponseError as e:
        # BUSYGROUP = already exists
        if "BUSYGROUP" in str(e):
            return
        raise


def b2s(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def b2i(x) -> int:
    return int(b2s(x))


def reset_group(r: redis.Redis, stream: str, group: str, start_id: bytes = b"$") -> None:
    try:
        r.xgroup_destroy(stream, group)
    except redis.ResponseError:
        pass
    ensure_group(r, stream, group, start_id=start_id)


# ---------------------------
# Producer: Webcam/RTSP -> Redis Stream (video:frames)
# ---------------------------

def producer(
    redis_url: str,
    frames_stream: str,
    cam: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cap_fps: Optional[int] = None,
    jpeg_quality: int = 80,
    max_stream_len: int = 2000,
    stop_event: Optional[threading.Event] = None,
):
    r = redis.Redis.from_url(redis_url, decode_responses=False)

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {cam}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if cap_fps:
        cap.set(cv2.CAP_PROP_FPS, float(cap_fps))

    print(f"[producer] camera={cam} size=({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}) fps={cap.get(cv2.CAP_PROP_FPS)}")

    frame_id = 0
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            ts_ns = time.time_ns()
            h, w = frame.shape[:2]
            jpeg = encode_jpeg(frame, quality=jpeg_quality)
            fields = {
                b"frame_id": str(frame_id).encode(),
                b"ts_ns": str(ts_ns).encode(),
                b"w": str(w).encode(),
                b"h": str(h).encode(),
                b"jpeg": jpeg,
            }
            r.xadd(frames_stream, fields, maxlen=max_stream_len, approximate=True)
            frame_id += 1
    finally:
        cap.release()
        print("[producer] stopped")


# ---------------------------
# Worker: Redis(video:frames) -> YOLO(track) -> Redis(video:results)
# ---------------------------

def worker(
    redis_url: str,
    frames_stream: str,
    results_stream: str,
    workers_group: str,
    consumer: str,
    model_path: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    tracker_cfg: str,
    calib_yaml: str,
    max_stream_len: int = 2000,
    block_ms: int = 1000,
    stop_event: Optional[threading.Event] = None,
):
    r = redis.Redis.from_url(redis_url, decode_responses=False)
    ensure_group(r, frames_stream, workers_group)
    ensure_group(r, results_stream, "renderers")  # create results stream ahead of time

    # Setup model
    det = YoloDetector(model_path, device=device)

    print(f"[worker] starting consumer={consumer} group={workers_group} model={model_path}")

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            resp = r.xreadgroup(
                groupname=workers_group,
                consumername=consumer,
                streams={frames_stream: b">"},
                count=1,
                block=block_ms,
            )
            if not resp:
                continue
            for _stream_name, messages in resp:
                for msg_id, fields in messages:
                    # Expect bytes keys; skip malformed entries (e.g., any legacy probe messages)
                    jpeg = fields.get(b"jpeg", None)
                    if jpeg is None:
                        r.xack(frames_stream, workers_group, msg_id)
                        continue
                    frame_id = b2i(fields.get(b"frame_id", b"0"))
                    ts_ns_in = b2i(fields.get(b"ts_ns", b"0"))
                    try:
                        frame = decode_jpeg_to_bgr(jpeg)
                    except Exception:
                        # Bad frame, ack and continue
                        r.xack(frames_stream, workers_group, msg_id)
                        continue

                    # Inference per-frame using predict (detection). This avoids Ultralytics .track generator limitations.
                    t0 = time.perf_counter()
                    results = det.model.predict(
                        source=frame,
                        imgsz=imgsz,
                        conf=conf,
                        device=device,
                        verbose=False,
                    )
                    r0 = results[0]
                    frame_orig = r0.orig_img if getattr(r0, "orig_img", None) is not None else frame
                    vis = r0.plot()

                    # Latency from Ultralytics speeds (ms) or fallback to elapsed
                    pipe_ms = None
                    try:
                        sp = getattr(r0, "speed", None)
                        if isinstance(sp, dict) and sp:
                            pipe_ms = sum(float(v) for v in sp.values() if v is not None)
                    except Exception:
                        pipe_ms = None
                    if pipe_ms is None:
                        pipe_ms = (time.perf_counter() - t0) * 1000.0

                    # Yaw/pitch overlay (no track IDs in detection mode)
                    boxes = getattr(r0, "boxes", None)
                    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                        for box in xyxy:
                            x1, y1, x2, y2 = box.astype(float)
                            u = 0.5 * (x1 + x2)
                            v = 0.5 * (y1 + y2)
                            try:
                                yaw, pitch = detection_center_to_angles(u, v, calib_yaml)
                                yaw_deg = np.degrees(yaw)
                                pitch_deg = np.degrees(pitch)
                                label = f"yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°"
                                org = (int(x1), max(0, int(y1) - 10))
                                cv2.putText(
                                    vis,
                                    label,
                                    org,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                            except Exception:
                                pass

                    # Overlay pipeline latency bottom-right
                    txt = f"{pipe_ms:.1f} ms"
                    (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    m = 10
                    x = int(vis.shape[1] - tw - m)
                    y = int(vis.shape[0] - m)
                    cv2.rectangle(vis, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
                    cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # Side-by-side
                    side = np.hstack([frame_orig, vis])
                    jpeg_vis = encode_jpeg(side, quality=85)
                    ts_ns_out = time.time_ns()

                    # Publish results and ack
                    fields_out = {
                        b"frame_id": str(frame_id).encode(),
                        b"ts_ns_in": str(ts_ns_in).encode(),
                        b"ts_ns_out": str(ts_ns_out).encode(),
                        b"w": str(side.shape[1]).encode(),
                        b"h": str(side.shape[0]).encode(),
                        b"jpeg": jpeg_vis,
                    }
                    r.xadd(results_stream, fields_out, maxlen=max_stream_len, approximate=True)
                    r.xack(frames_stream, workers_group, msg_id)
    finally:
        print("[worker] stopped")


# ---------------------------
# Renderer: Redis(video:results) -> display/save
# ---------------------------

def renderer(
    redis_url: str,
    results_stream: str,
    renderers_group: str,
    consumer: str,
    save: bool,
    fps_out: int,
    display_max_w: int,
    display_max_h: int,
    stop_event: Optional[threading.Event] = None,
):
    r = redis.Redis.from_url(redis_url, decode_responses=False)
    ensure_group(r, results_stream, renderers_group)

    writer = None
    out_path = None

    # Throttle for display
    fps = max(1, int(fps_out)) if fps_out and fps_out > 0 else 20
    target_dt = 1.0 / fps
    next_frame_time = time.perf_counter() + target_dt

    frames = 0

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            resp = r.xreadgroup(
                groupname=renderers_group,
                consumername=consumer,
                streams={results_stream: b">"},
                count=1,
                block=1000,
            )
            if not resp:
                continue
            for _stream_name, messages in resp:
                for msg_id, fields in messages:
                    jpeg = fields[b"jpeg"] if b"jpeg" in fields else fields["jpeg"]  # type: ignore
                    w = b2i(fields.get(b"w", b"0"))
                    h = b2i(fields.get(b"h", b"0"))
                    ts_ns_in = b2i(fields.get(b"ts_ns_in", b"0"))
                    ts_ns_out = b2i(fields.get(b"ts_ns_out", b"0"))

                    img = decode_jpeg_to_bgr(jpeg)

                    # Overlay end-to-end latency (capture -> render)
                    t_render_ns = time.time_ns()
                    e2e_ms = (t_render_ns - ts_ns_in) / 1e6 if ts_ns_in else 0.0
                    txt = f"E2E {e2e_ms:.1f} ms"
                    (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    m = 10
                    x = m
                    y = int(img.shape[0] - m)
                    cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
                    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

                    # Init writer lazily
                    if save and writer is None:
                        out_dir = Path("outputs")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        ts_tag = time.strftime("%Y%m%d_%H%M%S")
                        out_path = out_dir / f"stream_{ts_tag}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (img.shape[1], img.shape[0]))

                    if save and writer is not None:
                        writer.write(img)

                    # Display with fit-to-window like run_demo
                    try:
                        sh, sw = img.shape[:2]
                        scale = min(display_max_w / max(1, sw), display_max_h / max(1, sh), 1.0)
                        disp = img if scale >= 0.999 else cv2.resize(img, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
                        cv2.imshow("Drone tracking (stream)", disp)
                        now2 = time.perf_counter()
                        delay_ms = int(max(0.0, (next_frame_time - now2)) * 1000)
                        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
                        next_frame_time += target_dt
                        if key in (27, ord("q")):
                            if stop_event is not None:
                                stop_event.set()
                            r.xack(results_stream, renderers_group, msg_id)
                            raise KeyboardInterrupt
                    except Exception:
                        pass

                    r.xack(results_stream, renderers_group, msg_id)

                    frames += 1
                    if frames % 100 == 0:
                        print(f"[renderer] frames={frames}")
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"[renderer] Saved annotated video: {out_path}")
        cv2.destroyAllWindows()
        print("[renderer] stopped")


# ---------------------------
# CLI Entrypoint
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Streaming YOLO demo: Producer -> Worker -> Renderer (Redis Streams)")
    p.add_argument("--config", type=str, default="config/runtime.yaml", help="Runtime YAML config")
    p.add_argument("--model", type=str, default="models/export/yolov8n.engine", help="Path to .pt or TensorRT .engine model")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--tracker", type=str, default=None, help="Tracker config YAML")
    p.add_argument("--calib", type=str, default=None, help="Calibration YAML for yaw/pitch mapping")
    p.add_argument("--save", action="store_true", help="Save annotated MP4 under outputs/")
    p.add_argument("--fps_out", type=float, default=None, help="Display/record FPS target (renderer)")
    p.add_argument("--display_max_w", type=int, default=1600)
    p.add_argument("--display_max_h", type=int, default=900)

    # Streaming options
    p.add_argument("--role", type=str, choices=["producer", "worker", "renderer", "all"], default="all")
    p.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    p.add_argument("--frames_stream", type=str, default="video:frames")
    p.add_argument("--results_stream", type=str, default="video:results")
    p.add_argument("--workers_group", type=str, default="workers")
    p.add_argument("--renderers_group", type=str, default="renderers")
    p.add_argument("--consumer", type=str, default=f"c-{os.getpid()}")
    p.add_argument("--maxlen", type=int, default=2000, help="MAXLEN for streams (approximate)")
    p.add_argument("--start_from", type=str, choices=["latest", "begin"], default="latest", help="Where groups start reading when created/reset: latest ('$') or begin ('0')")
    p.add_argument("--reset_groups", dest="reset_groups", action="store_true", help="Destroy and recreate consumer groups before starting")
    p.add_argument("--keep_groups", dest="reset_groups", action="store_false", help="Destroy and recreate consumer groups before starting")
    p.add_argument("--purge_streams", dest="purge_streams", action="store_true", help="Delete streams before starting (clears history)")
    p.add_argument("--keep_streams", dest="purge_streams", action="store_false", help="Delete streams before starting (clears history)")

    # Producer camera options
    p.add_argument("--cam", type=int, default=0, help="Webcam index (OpenCV)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--cap_fps", type=int, default=None)
    p.add_argument("--jpeg_quality", type=int, default=80)
    
    p.set_defaults(
        reset_groups=True,
        purge_streams=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Load runtime config and merge with CLI
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
    fps_out = int(pick("fps_out", 20) or 20)
    start_from = args.start_from
    start_id = b"$" if start_from == "latest" else b"0"

    stop_event = threading.Event()

    def handle_sigint(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    # Optional purge/reset before launching roles
    if args.purge_streams or args.reset_groups:
        r_admin = redis.Redis.from_url(args.redis_url, decode_responses=False)
        if args.purge_streams:
            try:
                r_admin.delete(args.frames_stream)
            except Exception:
                pass
            try:
                r_admin.delete(args.results_stream)
            except Exception:
                pass
        if args.reset_groups:
            try:
                reset_group(r_admin, args.frames_stream, args.workers_group, start_id=start_id)
            except Exception:
                pass
            try:
                reset_group(r_admin, args.results_stream, args.renderers_group, start_id=start_id)
            except Exception:
                pass

    if args.role == "producer":
        producer(
            redis_url=args.redis_url,
            frames_stream=args.frames_stream,
            cam=args.cam,
            width=args.width,
            height=args.height,
            cap_fps=args.cap_fps,
            jpeg_quality=args.jpeg_quality,
            max_stream_len=args.maxlen,
            stop_event=stop_event,
        )
    elif args.role == "worker":
        worker(
            redis_url=args.redis_url,
            frames_stream=args.frames_stream,
            results_stream=args.results_stream,
            workers_group=args.workers_group,
            consumer=args.consumer,
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            tracker_cfg=tracker_cfg,
            calib_yaml=calib_yaml,
            max_stream_len=args.maxlen,
            stop_event=stop_event,
        )
    elif args.role == "renderer":
        renderer(
            redis_url=args.redis_url,
            results_stream=args.results_stream,
            renderers_group=args.renderers_group,
            consumer=args.consumer,
            save=bool(args.save),
            fps_out=fps_out,
            display_max_w=int(args.display_max_w),
            display_max_h=int(args.display_max_h),
            stop_event=stop_event,
        )
    else:
        # Run all three: producer + worker in threads; renderer in main (to own the UI loop)
        t_prod = threading.Thread(
            target=producer,
            kwargs=dict(
                redis_url=args.redis_url,
                frames_stream=args.frames_stream,
                cam=args.cam,
                width=args.width,
                height=args.height,
                cap_fps=args.cap_fps,
                jpeg_quality=args.jpeg_quality,
                max_stream_len=args.maxlen,
                stop_event=stop_event,
            ),
            daemon=True,
        )
        t_work = threading.Thread(
            target=worker,
            kwargs=dict(
                redis_url=args.redis_url,
                frames_stream=args.frames_stream,
                results_stream=args.results_stream,
                workers_group=args.workers_group,
                consumer=f"{args.consumer}-w",
                model_path=model_path,
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                tracker_cfg=tracker_cfg,
                calib_yaml=calib_yaml,
                max_stream_len=args.maxlen,
                stop_event=stop_event,
            ),
            daemon=True,
        )
        t_prod.start()
        t_work.start()
        try:
            renderer(
                redis_url=args.redis_url,
                results_stream=args.results_stream,
                renderers_group=args.renderers_group,
                consumer=f"{args.consumer}-r",
                save=bool(args.save),
                fps_out=fps_out,
                display_max_w=int(args.display_max_w),
                display_max_h=int(args.display_max_h),
                stop_event=stop_event,
            )
        finally:
            stop_event.set()
            t_prod.join(timeout=1.0)
            t_work.join(timeout=1.0)


if __name__ == "__main__":
    main()
