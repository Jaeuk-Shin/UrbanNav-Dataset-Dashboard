"""Frame extraction and metadata helpers for per-segment ``.mp4`` files."""

from __future__ import annotations

import base64

import cv2
import streamlit as st

VIDEO_FPS = 30.0
TEXTURE_MAX_WIDTH = 320


@st.cache_data(show_spinner=False)
def _video_frame_count(mp4_path: str) -> int | None:
    cap = cv2.VideoCapture(mp4_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def _video_aspect(mp4_path: str) -> float:
    cap = cv2.VideoCapture(mp4_path)
    try:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return float(w) / float(h) if w > 0 and h > 0 else 16.0 / 9.0
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def _frame_jpeg_b64(mp4_path: str, vid_idx: int,
                     max_w: int = TEXTURE_MAX_WIDTH) -> str | None:
    """Extract one MP4 frame, return base64-encoded JPEG (no data URI prefix)."""
    cap = cv2.VideoCapture(mp4_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(vid_idx)))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    finally:
        cap.release()


def _video_frame_to_pose_idx(vid_frame: int, n_pose: int,
                             n_video: int | None) -> int:
    """Map video frame index (native fps) to pose row index."""
    if n_video and n_video > 0:
        return min(int(vid_frame * n_pose / n_video), n_pose - 1)
    return min(vid_frame // 6, n_pose - 1)
