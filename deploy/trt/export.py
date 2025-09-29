import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to TensorRT engine")
    p.add_argument("--config", type=str, default="config/runtime.yaml", help="Runtime YAML config")
    p.add_argument("--model", type=str, default=None, help="Path to .pt model to export")
    p.add_argument("--imgsz", type=int, default=None, help="Export image size")
    p.add_argument("--half", action="store_true", help="Export FP16 engine")
    p.add_argument("--int8", action="store_true", help="Export INT8 engine (requires calibrator)")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes (lowest latency prefers fixed)")
    p.add_argument("--device", type=str, default=None, help="CUDA device id or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    # Load config and merge with CLI (CLI overrides when provided)
    cfg = {}
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    def pick(key, default=None):
        val = getattr(args, key, None)
        if val in (None,):
            return cfg.get(key, default)
        return val

    model_path = pick("model", "yolov8n.pt")
    imgsz = int(pick("imgsz", 640))
    device = str(pick("device", "0"))

    model = YOLO(
        model_path,
        task="detect",
    )
    out_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=bool(args.half),
        int8=bool(args.int8),
        dynamic=bool(args.dynamic),
        device=device,
    )
    out_path = Path(out_path)
    export_dir = Path("models/export")
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / out_path.name
    if out_path.resolve() != target.resolve():
        try:
            target.write_bytes(out_path.read_bytes())
        except Exception:
            pass
    print(f"[deploy.trt.export] Engine: {target.as_posix() if target.exists() else out_path.as_posix()}")


if __name__ == "__main__":
    main()
