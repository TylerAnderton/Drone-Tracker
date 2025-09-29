import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate camera and write config/calib.yaml")
    p.add_argument("--images", type=str, required=True, help="Directory with chessboard images or a glob pattern")
    p.add_argument("--pattern", type=str, default="*.jpg", help="Pattern when --images is a directory")
    p.add_argument("--rows", type=int, required=True, help="Internal corners per row (width direction)")
    p.add_argument("--cols", type=int, required=True, help="Internal corners per column (height direction)")
    p.add_argument("--square", type=float, required=True, help="Square size in meters (or any unit)")
    p.add_argument("--output", type=str, default="config/calib.yaml", help="Output YAML path")
    return p.parse_args()


def collect_images(images: str, pattern: str):
    p = Path(images)
    if p.is_dir():
        files = sorted(list(p.glob(pattern)))
    else:
        files = sorted([Path(x) for x in Path().glob(images)])
    return [f for f in files if f.is_file()]


def main():
    args = parse_args()
    images = collect_images(args.images, args.pattern)
    if not images:
        raise FileNotFoundError(f"No images found for {args.images}")

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.rows, 0:args.cols].T.reshape(-1, 2)
    objp *= float(args.square)

    objpoints, imgpoints = [], []
    gray_shape = None

    for f in images:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (args.rows, args.cols), None)
        if ret:
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objpoints.append(objp)
            imgpoints.append(corners2)

    if not objpoints:
        raise RuntimeError("No chessboard corners detected. Check rows/cols and image quality.")

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    print(f"[cli_calibrate] RMS reprojection error: {ret:.4f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"K": K.tolist(), "D": D.ravel().tolist(), "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    with open(out, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[cli_calibrate] Wrote calibration to {out}")


if __name__ == "__main__":
    main()
