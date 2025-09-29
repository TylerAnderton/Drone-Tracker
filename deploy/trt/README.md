# TensorRT Deploy

- Export FP16 engine (preferred on RTX 4070 SUPER):
```bash
python -m deploy.trt.export --half
# -> models/export/yolov8n.engine
```

- Optional INT8 with calibration images (prepare a directory of Anti-UAV frames):
```bash
python -m deploy.trt.export --int8
```

- Alternative: trtexec (when you want full control):
```bash
trtexec --onnx=models/export/yolov8n.onnx \
        --saveEngine=models/export/yolov8n_fp16.engine \
        --fp16 --workspace=2048 --shapes=input:1x3x640x640
```
