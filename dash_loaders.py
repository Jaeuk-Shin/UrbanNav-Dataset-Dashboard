"""Cached data-loading helpers shared across queries."""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# Approximate dataset frame rate (~30 Hz camera sampled every ~15 frames).
DATASET_FPS = 2.0


@st.cache_data
def load_json(path: str):
    """Load and cache a JSON file (keyed by path string)."""
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_poses(path: str) -> np.ndarray:
    """Load a whitespace-delimited pose file.

    Columns: idx  frame_id  x  y  z  qx  qy  qz  qw
    """
    return np.loadtxt(path)


@st.cache_data
def load_segment_cache(data_root: str) -> dict | None:
    """Load ``.segment_cache.pkl`` produced by ``annotate.py``, if present.

    Returns the raw cache dict with keys ``"format"``, ``"data"``, etc.,
    or *None* when no cache file exists.
    """
    p = Path(data_root) / ".segment_cache.pkl"
    if not p.exists():
        return None
    try:
        return pickle.loads(p.read_bytes())
    except Exception:
        return None


@st.cache_data
def list_segments(data_root: str) -> list:
    """Return sorted segment names.

    Prefers ``.segment_cache.pkl`` when available; otherwise falls back to
    scanning ``rgb/`` for directories (frame-based) or ``.mp4`` files
    (video-based).
    """
    cache = load_segment_cache(data_root)
    if cache is not None:
        return sorted(cache.get("data", {}).keys())
    rgb_dir = Path(data_root) / "rgb"
    if not rgb_dir.is_dir():
        return []
    names: set[str] = set()
    for entry in rgb_dir.iterdir():
        if entry.is_dir():
            names.add(entry.name)
        elif entry.suffix.lower() == ".mp4":
            names.add(entry.stem)
    return sorted(names)


def segment_frame_count(data_root: str, seg: str) -> int | None:
    """Return the number of frames for *seg* from the cache, or *None*."""
    cache = load_segment_cache(data_root)
    if cache is None:
        return None
    entry = cache.get("data", {}).get(seg)
    if entry is None:
        return None
    if isinstance(entry, tuple):
        # video format: (video_path, [frame_indices])
        return len(entry[1])
    if isinstance(entry, list):
        # jpg format: [path_strings]
        return len(entry)
    return None
