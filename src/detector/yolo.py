from pathlib import Path
from typing import Iterator, Optional
from ultralytics import YOLO


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLO that supports .pt and TensorRT .engine
    and exposes a unified track() generator for video sources.
    """

    def __init__(self, model_path: str, device: str = "0"):
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)

    def warmup(self, source: str, imgsz: int = 640, conf: float = 0.25):
        try:
            _ = self.model.predict(
                source=Path(source).as_posix(),
                imgsz=imgsz,
                conf=conf,
                device=self.device,
                stream=False,
                verbose=False,
                max_det=100,
                vid_stride=30,
            )
        except Exception:
            pass

    def track(
        self,
        source: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker_cfg: str = "config/bytetrack.yaml",
        save: bool = False,
        persist: bool = True,
    ) -> Iterator:
        return self.model.track(
            source=Path(source).as_posix(),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            tracker=tracker_cfg,
            stream=True,
            save=save,
            verbose=False,
            persist=persist,
        )
