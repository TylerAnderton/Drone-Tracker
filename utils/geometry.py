import numpy as np
import yaml
from typing import Tuple


def load_calibration(yaml_path: str):
    """Load camera intrinsics/extrinsics from a YAML file.

    Expected keys: K (3x3), D (distortion, len=5 or len=8), R (3x3), t (3,)
    Any missing keys default to identity/zeros.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    K = np.array(cfg.get("K", np.eye(3)), dtype=np.float64)
    D = np.array(cfg.get("D", np.zeros(5)), dtype=np.float64).reshape(-1)
    R = np.array(cfg.get("R", np.eye(3)), dtype=np.float64)
    t = np.array(cfg.get("t", np.zeros(3)), dtype=np.float64).reshape(3)
    return K, D, R, t


def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Back-project pixel (u,v) to a unit direction ray in the camera frame.
    K is 3x3 intrinsic matrix.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    r_cam = np.array([x, y, 1.0], dtype=np.float64)
    r_cam = r_cam / np.linalg.norm(r_cam)
    return r_cam


def cam_to_turret(ray_cam: np.ndarray, R_cam_turret: np.ndarray) -> np.ndarray:
    """Transform a direction vector from camera frame to turret frame.
    R_cam_turret rotates camera-frame vector into turret frame.
    """
    return R_cam_turret @ ray_cam


def ray_to_yaw_pitch(ray_t: np.ndarray) -> Tuple[float, float]:
    """Compute yaw/pitch from a direction vector in turret frame.
    Yaw = atan2(x, z), Pitch = atan2(-y, sqrt(x^2 + z^2)). Returns radians.
    """
    x, y, z = ray_t.astype(np.float64)
    yaw = np.arctan2(x, z)
    pitch = np.arctan2(-y, np.sqrt(x * x + z * z))
    return float(yaw), float(pitch)
