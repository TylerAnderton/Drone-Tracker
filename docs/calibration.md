# Calibration

- **Intrinsics**: estimate camera matrix K and distortion D with a chessboard.
- **Extrinsics**: default to identity; align camera to turret later.
- **Pixel → Ray → Yaw/Pitch**:
  - r_cam = K^{-1} [u, v, 1]^T normalized
  - r_t = R_cam_turret · r_cam
  - yaw = atan2(x, z), pitch = atan2(-y, sqrt(x^2+z^2))
- **Procedure**:
  1. Capture 15–25 chessboard images at varied poses.
  2. Run `src/calib/cli_calibrate.py` to produce `config/calib.yaml`.
  3. Validate by projecting a few known points; check reprojection error.
