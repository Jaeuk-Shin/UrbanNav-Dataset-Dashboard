"""Segment discovery with caching."""

import hashlib
import os
import pickle
from fnmatch import fnmatch
from pathlib import Path

import cv2
from tqdm import tqdm

from pipeline.frames import FrameRef


def _video_frame_refs(video_path, fps):
    """Build a list of FrameRefs for *fps*-subsampled frames."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total <= 0:
        return []
    # CAP_PROP_FRAME_COUNT often overcounts by a few frames (container
    # metadata vs actual decodeable frames).  Subtract 1 second of frames
    # to avoid requesting unreadable tail frames.
    safe_total = max(total - int(video_fps), 1)
    duration = safe_total / video_fps
    step = 1.0 / fps
    refs = []
    t = 0.0
    while t < duration:
        idx = min(int(t * video_fps), total - 1)
        refs.append(FrameRef(video_path, idx))
        t += step
    return refs


def _segment_cache_key(rgb_root, input_format, fps, subsample):
    """Hash (parameters + directory listing) for cache invalidation."""
    try:
        names = sorted(os.listdir(rgb_root))
    except FileNotFoundError:
        names = []
    h = hashlib.sha256()
    h.update(str(Path(rgb_root).resolve()).encode())
    h.update(f"{input_format}|{fps}|{subsample}".encode())
    for n in names:
        h.update(n.encode())
    return h.hexdigest()


def _serialize_segments(segments, input_format):
    """Convert segments dict to a compact pickle-friendly format."""
    if input_format == "video":
        return {
            name: (str(frames[0].video_path), [f.frame_idx for f in frames])
            for name, frames in segments.items()
        }
    return {name: [str(p) for p in frames] for name, frames in segments.items()}


def _deserialize_segments(data, input_format):
    """Reconstruct segments dict from compact format."""
    if input_format == "video":
        segments = {}
        for name, (vpath, indices) in data.items():
            vp = Path(vpath)
            segments[name] = [FrameRef(vp, idx) for idx in indices]
        return segments
    return {name: [Path(p) for p in paths] for name, paths in data.items()}


def _discover_fresh(rgb_root, subsample, fps, input_format):
    """Scan rgb/ and return the full unfiltered segments dict."""
    segments = {}
    items = sorted(rgb_root.iterdir())
    for item in tqdm(items, desc="discover", unit="seg", leave=False):
        if input_format == "jpg":
            if not item.is_dir():
                continue
            name = item.name
            frames = sorted(item.glob("*.jpg"))
            if subsample > 1:
                frames = frames[::subsample]
        elif input_format == "video":
            if not item.is_file() or item.suffix.lower() != ".mp4":
                continue
            name = item.stem
            frames = _video_frame_refs(item, fps or 1.0)
        else:
            raise ValueError(f"Unknown input format: {input_format!r}")
        if frames:
            segments[name] = frames
    return segments


def discover_segments(
        data_root, 
        pattern=None, 
        names=None, 
        subsample=1,
        fps=None, 
        input_format="jpg"
    ):
    """Return {segment_name: [frame Paths or FrameRefs]}.

    Results are cached to ``{data_root}/.segment_cache.pkl`` and
    auto-invalidated when the ``rgb/`` directory contents or parameters
    change.  Delete the cache file to force re-discovery.

    *input_format* selects the data layout under ``rgb/``:
      - ``"jpg"``   — each segment is a sub-directory of JPEG frames
      - ``"video"`` — each segment is an ``.mp4`` file

    *subsample* keeps every Nth JPG frame; *fps* sets the video extraction
    rate (default 1.0 fps).
    """
    rgb_root = Path(data_root) / "rgb"
    cache_file = Path(data_root) / ".segment_cache.pkl"
    cache_key = _segment_cache_key(rgb_root, input_format, fps, subsample)

    # Try cache
    segments = None
    if cache_file.exists():
        try:
            cached = pickle.loads(cache_file.read_bytes())
            if cached.get("key") == cache_key:
                segments = _deserialize_segments(
                    cached["data"], cached["format"]
                )
                tqdm.write(
                    f"  Loaded segment cache ({len(segments)} segments)"
                )
        except Exception:
            pass

    # Cache miss — full discovery
    if segments is None:
        segments = _discover_fresh(rgb_root, subsample, fps, input_format)
        try:
            cache_file.write_bytes(pickle.dumps({
                "key": cache_key,
                "format": input_format,
                "data": _serialize_segments(segments, input_format),
            }))
            tqdm.write(f"  Cached {len(segments)} segments → {cache_file}")
        except Exception:
            pass

    # Apply filters
    if pattern or names:
        name_set = set(names) if names else None
        segments = {
            k: v for k, v in segments.items()
            if (not pattern or fnmatch(k, pattern))
            and (not name_set or k in name_set)
        }

    return segments


def total_frames(segments):
    return sum(len(v) for v in segments.values())
