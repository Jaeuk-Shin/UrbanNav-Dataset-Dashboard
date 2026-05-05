"""Browse valid frames from segments that pass curation filters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from dash_query import Query
from dash_types import QueryOutput, FrameResult

from curation.database import get_connection


@st.cache_data(ttl=3600, show_spinner="Loading filtered segments...")
def _load_filtered_segments(db_path: str, _mtime: float):
    """Load segment names, pass rates, and valid frame indices from DB."""
    conn = get_connection(db_path, readonly=True)
    rows = conn.execute(
        """SELECT s.name, s.num_frames, s.split, f.valid_mask
           FROM segments s
           JOIN segment_filter_data f ON s.segment_id = f.segment_id"""
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        mask = np.frombuffer(r["valid_mask"], dtype=np.uint8).astype(bool)
        valid_count = int(mask.sum())
        if r["num_frames"] == 0:
            continue
        pass_rate = 100.0 * valid_count / r["num_frames"]
        valid_indices = np.where(mask)[0].tolist()
        result.append({
            "segment": r["name"],
            "num_frames": r["num_frames"],
            "valid_count": valid_count,
            "pass_rate": pass_rate,
            "split": r["split"] or "(none)",
            "valid_indices": valid_indices,
        })
    return result


class FilteredBrowse(Query):
    name = "Filtered Browse"
    description = "Browse valid frames from segments passing curation filters"

    def build_params(self):
        min_pr = st.sidebar.slider("Min pass rate (%)", 0, 100, 50,
                                   key="fb_min_pass")
        split = st.sidebar.selectbox("Split", ["all", "train", "val", "test"],
                                     key="fb_split")
        sort_by = st.sidebar.selectbox(
            "Sort by",
            ["pass_rate", "valid_count", "segment"],
            key="fb_sort",
        )
        valid_only = st.sidebar.checkbox("Show only valid frames", True,
                                         key="fb_valid_only")
        frames_per_seg = st.sidebar.slider(
            "Frames per segment", 1, 10, 3, key="fb_frames_per_seg")
        return {
            "min_pass_rate": min_pr,
            "split": split,
            "sort_by": sort_by,
            "valid_only": valid_only,
            "frames_per_seg": frames_per_seg,
        }

    def execute(self, root, segments, params):
        db_path = params.get("db_path", "")
        if not db_path or not Path(db_path).exists():
            return QueryOutput([], "table", "Filtered Browse",
                               "No curation DB found.")

        mtime = os.path.getmtime(db_path)
        all_segs = _load_filtered_segments(db_path, mtime)

        # Apply filters
        min_pr = params["min_pass_rate"]
        split = params["split"]
        filtered = [s for s in all_segs if s["pass_rate"] >= min_pr]
        if split != "all":
            filtered = [s for s in filtered if s["split"] == split]

        # Sort
        sort_key = params["sort_by"]
        reverse = sort_key != "segment"
        filtered.sort(key=lambda s: s[sort_key], reverse=reverse)

        # Build frame results: pick evenly-spaced frames from each segment
        valid_only = params["valid_only"]
        fps = params["frames_per_seg"]
        results = []
        for seg_info in filtered:
            seg = seg_info["segment"]
            if valid_only:
                indices = seg_info["valid_indices"]
            else:
                indices = list(range(seg_info["num_frames"]))

            if not indices:
                continue

            # Pick evenly spaced frames
            if len(indices) <= fps:
                picked = indices
            else:
                step = len(indices) / fps
                picked = [indices[int(i * step)] for i in range(fps)]

            for idx in picked:
                frame_id = f"{idx:06d}"
                results.append(FrameResult(
                    seg, frame_id,
                    score=seg_info["pass_rate"],
                    metadata={
                        "caption": (f"{seg} f{idx}  |  "
                                    f"{seg_info['pass_rate']:.0f}% valid  |  "
                                    f"{seg_info['split']}"),
                    },
                ))

        desc = (f"{len(filtered)} segments with pass rate >= {min_pr}%"
                + (f" (split={split})" if split != "all" else ""))
        return QueryOutput(results, "image_grid", "Filtered Browse", desc)
