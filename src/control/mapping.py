from typing import Tuple
import numpy as np
from utils.geometry import load_calibration, pixel_to_ray, cam_to_turret, ray_to_yaw_pitch


def detection_center_to_angles(u: float, v: float, calib_yaml: str) -> Tuple[float, float]:
    """Return yaw, pitch (radians) for a detection center pixel (u,v)."""
    K, D, R, t = load_calibration(calib_yaml)
    r_cam = pixel_to_ray(u, v, K)
    r_t = cam_to_turret(r_cam, R)
    return ray_to_yaw_pitch(r_t)
