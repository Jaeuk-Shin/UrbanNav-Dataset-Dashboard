"""Pose loading + smoothing helpers shared across the curation pipeline.

Three functions:

* :func:`load_pose_from_text` parses a ``pose/*.txt`` file into an ``(N, 7)``
  ``float64`` array, dropping the leading frame-index column when present
  and truncating at the first NaN row.
* :func:`load_pose_from_blob` deserialises the binary BLOB stored in
  ``segment_poses.pose_data`` (already shaped as 7 columns).
* :func:`smooth_window` applies an edge-padded uniform sliding-window
  average that preserves the input length — used by every filter metric.
"""

from __future__ import annotations

import numpy as np


def load_pose_from_text(path: str) -> np.ndarray:
    """Parse a pose text file into an (N, 7) float64 array.

    Drops the frame-index column (column 0) and truncates at the first NaN
    row.  Returns ``(0, 7)`` when the file has no valid rows.
    """
    raw = np.loadtxt(path)
    if raw.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    pose = raw[:, 1:] if raw.shape[1] == 8 else raw
    nan_mask = np.isnan(pose).any(axis=1)
    if nan_mask.any():
        pose = pose[: np.argmax(nan_mask)]
    return pose.astype(np.float64)


def load_pose_from_blob(blob: bytes | None) -> np.ndarray:
    """Deserialise an ``(N, 7)`` float64 pose array from a binary BLOB."""
    if not blob:
        return np.empty((0, 7), dtype=np.float64)
    return np.frombuffer(blob, dtype=np.float64).reshape(-1, 7)


def smooth_window(arr: np.ndarray, half_window: int) -> np.ndarray:
    """Uniform sliding-window average, edge-padded to preserve length."""
    if half_window <= 0 or len(arr) == 0:
        return arr.copy()
    kernel_size = 2 * half_window + 1
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.pad(arr, half_window, mode="edge")
    return np.convolve(padded, kernel, mode="valid")
