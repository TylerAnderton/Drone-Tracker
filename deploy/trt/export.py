import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to TensorRT engine")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt model to export")
    p.add_argument("--imgsz", type=int, default=640, help="Export image size")
    p.add_argument("--half", action="store_true", help="Export FP16 engine")
    p.add_argument("--int8", action="store_true", help="Export INT8 engine (requires calibrator)")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes (lowest latency prefers fixed)")
    p.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    out_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        device=args.device,
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
