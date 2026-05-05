"""Shared helpers for visualizers."""

from pathlib import Path

import cv2

# 12-colour palette for overlays and bounding boxes.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]


def load_rgb(root: Path, seg: str, fid: str):
    """Load an RGB frame as (H, W, 3) uint8 array, or *None*.

    Tries the frame-directory layout first (``rgb/{seg}/{fid}.jpg``),
    then falls back to extracting the frame from a video file
    (``rgb/{seg}.mp4``) using the frame index encoded in *fid*.
    """
    # Frame-directory layout
    p = root / "rgb" / seg / f"{fid}.jpg"
    if p.exists():
        img = cv2.imread(str(p))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    # Video-file layout
    vid = root / "rgb" / f"{seg}.mp4"
    if not vid.exists():
        return None
    cap = cv2.VideoCapture(str(vid))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
