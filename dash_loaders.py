"""Cached data-loading helpers shared across queries."""

import json
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
def list_segments(rgb_dir: str) -> list:
    """Return sorted segment directory names under *rgb_dir*."""
    return sorted(d.name for d in Path(rgb_dir).iterdir() if d.is_dir())
