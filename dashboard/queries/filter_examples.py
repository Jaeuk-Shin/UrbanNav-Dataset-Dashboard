"""Browse thumbnails of frames rejected by a specific curation filter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from ..query import Query
from ..types import QueryOutput, FrameResult

from curation.database import get_connection
from curation.filters import FilterConfig, compute_filter_masks


# The 10 curation filters. Names match `compute_filter_masks` keys
# (plus `stop_without_reasons`, inferred from combined vs AND-of-others).
FILTER_CHOICES = [
    "forward_camera_angle",
    "roll_change",
    "pitch_change",
    "yaw_change",
    "abs_pitch",
    "abs_roll",
    "velocity_spike",
    "height_change",
    "sustained_slow",
    "stop_without_reasons",
]

# Metric BLOB column used to annotate each rejected frame's caption.
_METRIC_FOR_FILTER = {
    "forward_camera_angle": "forward_camera_angles",
    "roll_change": "roll_changes",
    "pitch_change": "pitch_changes",
    "yaw_change": "yaw_changes",
    "abs_pitch": "abs_pitch",
    "abs_roll": "abs_roll",
    "velocity_spike": "velocities",
    "height_change": "height_changes",
    "sustained_slow": "velocities",
    "stop_without_reasons": "velocities",
}

_METRIC_KEYS = [
    "velocities", "forward_camera_angles",
    "roll_changes", "pitch_changes", "yaw_changes",
    "abs_pitch", "abs_roll", "height_changes",
]


@st.cache_data(ttl=3600, show_spinner="Scanning filter rejections...")
def _scan_rejections(db_path: str, filter_name: str,
                     max_segments: int, _mtime: float):
    """Per-segment rejected-frame indices + metric values for a single filter."""
    conn = get_connection(db_path, readonly=True)
    rows = conn.execute(
        """SELECT s.name, s.num_frames,
                  f.velocities, f.forward_camera_angles,
                  f.roll_changes, f.pitch_changes, f.yaw_changes,
                  f.abs_pitch, f.abs_roll, f.height_changes,
                  f.valid_mask
           FROM segments s
           JOIN segment_filter_data f ON s.segment_id = f.segment_id
           LIMIT ?""",
        (max_segments,),
    ).fetchall()
    conn.close()

    cfg = FilterConfig()
    metric_col = _METRIC_FOR_FILTER[filter_name]
    out = []
    for r in rows:
        nf = r["num_frames"]
        if nf == 0:
            continue

        metrics: dict[str, np.ndarray] = {}
        for k in _METRIC_KEYS:
            blob = r[k]
            metrics[k] = (np.frombuffer(blob, dtype=np.float32)
                          if blob else np.zeros(nf, dtype=np.float32))

        ind = compute_filter_masks(metrics, cfg)
        combined = np.frombuffer(r["valid_mask"], dtype=np.uint8).astype(bool)

        if filter_name == "stop_without_reasons":
            and_others = np.logical_and.reduce(list(ind.values()))
            rejected = np.where(and_others & ~combined)[0]
        else:
            rejected = np.where(~ind[filter_name])[0]

        if rejected.size == 0:
            continue

        mvals = metrics.get(metric_col, np.zeros(nf, dtype=np.float32)).tolist()
        out.append({
            "segment": r["name"],
            "rejected": rejected.tolist(),
            "metric_values": mvals,
            "num_frames": nf,
        })
    return out


class FilterExamples(Query):
    name = "Filter Examples"
    description = "Thumbnails of frames rejected by a specific curation filter"

    def build_params(self):
        filt = st.sidebar.selectbox("Filter", FILTER_CHOICES, key="fx_filter")
        max_segs = st.sidebar.slider(
            "Segments to scan", 50, 5000, 500, step=50, key="fx_max_segs")
        per_seg = st.sidebar.slider(
            "Frames per segment", 1, 10, 2, key="fx_per_seg")
        sort_order = st.sidebar.selectbox(
            "Sort", ["worst first", "best first", "segment order"],
            key="fx_sort")
        return {
            "filter": filt,
            "max_segments": max_segs,
            "frames_per_seg": per_seg,
            "sort_order": sort_order,
        }

    def execute(self, root, segments, params):
        db_path = params.get("db_path", "")
        if not db_path or not Path(db_path).exists():
            return QueryOutput([], "table", "Filter Examples",
                               "No curation DB found.")

        mtime = os.path.getmtime(db_path)
        filter_name = params["filter"]
        max_segs = params["max_segments"]
        per_seg = params["frames_per_seg"]
        sort_order = params["sort_order"]

        data = _scan_rejections(db_path, filter_name, max_segs, mtime)

        # For wall-clock filters (stop_without_reasons), the metric value
        # (velocity) is informational only; don't use it as a severity score.
        severity_is_metric = filter_name not in ("stop_without_reasons",
                                                  "sustained_slow")

        results: list[FrameResult] = []
        for seg_info in data:
            seg = seg_info["segment"]
            rejected = seg_info["rejected"]
            mvals = seg_info["metric_values"]

            # Pick evenly-spaced rejected frames so a single busy segment
            # doesn't drown out the rest.
            if len(rejected) <= per_seg:
                picked = rejected
            else:
                step = len(rejected) / per_seg
                picked = [rejected[int(i * step)] for i in range(per_seg)]

            for idx in picked:
                mv = mvals[idx] if idx < len(mvals) else 0.0
                caption = f"{seg}  f{idx}  |  {filter_name}={mv:.3f}"
                score = float(mv) if severity_is_metric else 0.0
                results.append(FrameResult(
                    seg, f"{idx:06d}",
                    score=score,
                    metadata={
                        "caption": caption,
                        "tags": [filter_name],
                    },
                ))

        if sort_order == "worst first":
            results.sort(key=lambda r: r.score, reverse=True)
        elif sort_order == "best first":
            results.sort(key=lambda r: r.score)
        # else keep insertion order (segment order)

        desc = (f"Filter '{filter_name}': {len(results)} rejected-frame examples "
                f"from {len(data)} segments (scanned up to {max_segs})")
        return QueryOutput(results, "image_grid", "Filter Examples", desc)
