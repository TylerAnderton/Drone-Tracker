import argparse
from pathlib import Path
import cv2
import re
import glob as _glob


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    p = argparse.ArgumentParser(description="Convert a directory of frames into an MP4 video")
    p.add_argument("--input", type=str, required=True, help="Path to directory with frames or a glob pattern")
    p.add_argument("--output", type=str, default="", help="Output .mp4 path; defaults to outputs/<dirname>.mp4")
    p.add_argument("--fps", type=int, default=30, help="Output FPS")
    p.add_argument("--pattern", type=str, default="*.jpg", help="Filename pattern when input is a directory")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)

    # Collect frame files
    if in_path.is_dir():
        frames = sorted([p for p in in_path.glob(args.pattern)], key=lambda p: natural_key(p.name))
        basename = in_path.name
    else:
        # Absolute or relative glob pattern
        matches = _glob.glob(args.input)
        frames = sorted([Path(p) for p in matches], key=lambda p: natural_key(p.name))
        basename = Path(args.input).stem

    if not frames:
        raise FileNotFoundError(f"No frames found for input={args.input}")

    # Read first frame for size
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    h, w = first.shape[:2]

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else (out_dir / f"{basename}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {out_file}")

    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"[frames_to_video] Wrote {len(frames)} frames -> {out_file}")


if __name__ == "__main__":
    main()
