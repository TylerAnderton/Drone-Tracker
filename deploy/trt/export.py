import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil


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
    weights_dir = Path("models/weights")
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Move final engine to models/export
    target = export_dir / out_path.name
    if out_path.resolve() != target.resolve():
        # Prefer move to keep root clean; fallback to copy+remove
        try:
            out_path.replace(target)
        except Exception:
            try:
                shutil.copy2(out_path, target)
                try:
                    out_path.unlink()
                except Exception:
                    pass
            except Exception:
                pass

    # Also move common side artifacts created by export or download into proper folders
    stem = out_path.stem  # e.g., yolov8n
    root = Path.cwd()
    onnx_src = root / f"{stem}.onnx"
    pt_src = root / f"{stem}.pt"
    onnx_dst = export_dir / onnx_src.name
    pt_dst = weights_dir / pt_src.name
    # Move ONNX if created in root
    if onnx_src.exists() and onnx_src.resolve() != onnx_dst.resolve():
        try:
            onnx_src.replace(onnx_dst)
        except Exception:
            try:
                shutil.copy2(onnx_src, onnx_dst)
                try:
                    onnx_src.unlink()
                except Exception:
                    pass
            except Exception:
                pass
    # Move downloaded PT checkpoint (for alias models like 'yolov8n.pt') to models/weights
    if pt_src.exists() and pt_src.resolve() != pt_dst.resolve():
        try:
            pt_src.replace(pt_dst)
        except Exception:
            try:
                shutil.copy2(pt_src, pt_dst)
                try:
                    pt_src.unlink()
                except Exception:
                    pass
            except Exception:
                pass

    print(f"[deploy.trt.export] Engine: {target.as_posix() if target.exists() else out_path.as_posix()}")


if __name__ == "__main__":
    main()
