# TensorRT Deploy

- Export FP16 engine (preferred on RTX 4070 SUPER):
```bash
python -m deploy.trt.export --model yolov8n.pt --imgsz 640 --half --device 0
# -> models/export/yolov8n.engine
```

- Optional INT8 with calibration images (prepare a directory of Anti-UAV frames):
```bash
python -m deploy.trt.export --model yolov8n.pt --imgsz 640 --int8 --device 0
```

- Alternative: trtexec (when you want full control):
```bash
trtexec --onnx=models/export/yolov8n.onnx \
        --saveEngine=models/export/yolov8n_fp16.engine \
        --fp16 --workspace=2048 --shapes=input:1x3x640x640
```
