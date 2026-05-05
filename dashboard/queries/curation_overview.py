"""Aggregate curation-filter statistics across the full dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from ..query import Query
from ..types import QueryOutput, SegmentResult

# Reuse curation internals for filter re-decomposition
from curation.database import get_connection
from curation.filters import FilterConfig, compute_filter_masks


_METRIC_KEYS = (
    "velocities", "forward_camera_angles",
    "roll_changes", "pitch_changes", "yaw_changes",
    "abs_pitch", "abs_roll", "height_changes",
)


def _decode_metrics(row, num_frames: int) -> dict[str, np.ndarray]:
    """Deserialise per-frame metric BLOBs into ``(N,) float32`` arrays."""
    out: dict[str, np.ndarray] = {}
    for key in _METRIC_KEYS:
        blob = row[key]
        out[key] = (np.frombuffer(blob, dtype=np.float32) if blob
                    else np.zeros(num_frames, dtype=np.float32))
    return out


@st.cache_data(ttl=3600, show_spinner="Computing per-filter breakdown...")
def _compute_aggregate(db_path: str, _mtime: float):
    """Compute aggregate filter stats.  Cached by DB path + mtime."""
    conn = get_connection(db_path, readonly=True)
    cfg = FilterConfig()

    # Load all segments with filter data
    rows = conn.execute(
        """SELECT s.segment_id, s.name, v.name AS video,
                  s.num_frames, f.valid_mask,
                  f.velocities, f.forward_camera_angles,
                  f.roll_changes, f.pitch_changes, f.yaw_changes,
                  f.abs_pitch, f.abs_roll, f.height_changes
           FROM segments s
           JOIN segment_filter_data f ON s.segment_id = f.segment_id
           JOIN videos v ON s.video_id = v.video_id"""
    ).fetchall()

    total_segments = len(rows)
    total_frames = 0
    valid_frames = 0
    alive = 0
    dead = 0
    per_filter: dict[str, int] = {}
    pass_rates: list[float] = []
    video_stats: dict[str, dict] = {}
    segment_rows: list[dict] = []

    for r in rows:
        nf = r["num_frames"]
        mask = np.frombuffer(r["valid_mask"], dtype=np.uint8).astype(bool)
        vf = int(mask.sum())
        pr = 100.0 * vf / nf if nf > 0 else 0.0

        total_frames += nf
        valid_frames += vf
        pass_rates.append(pr)
        if vf > 0:
            alive += 1
        else:
            dead += 1

        # Per-filter decomposition (filters 1-9 + sustained_slow; stop_without_reasons
        # is DB-annotation-dependent and not counted here).
        filt_masks = compute_filter_masks(_decode_metrics(r, nf), cfg)
        for k, m in filt_masks.items():
            per_filter[k] = per_filter.get(k, 0) + int(np.sum(~m))

        # Per-video accumulation
        vid = r["video"]
        vs = video_stats.setdefault(vid, {
            "video": vid, "segments": 0, "alive": 0, "dead": 0,
            "total_frames": 0, "valid_frames": 0,
        })
        vs["segments"] += 1
        vs["alive"] += 1 if vf > 0 else 0
        vs["dead"] += 1 if vf == 0 else 0
        vs["total_frames"] += nf
        vs["valid_frames"] += vf

        segment_rows.append({
            "segment": r["name"], "video": vid,
            "num_frames": nf, "valid_frames": vf,
            "pass_rate": round(pr, 1),
        })

    conn.close()

    # Pass-rate histogram (20 bins, 0-100%)
    hist, _ = np.histogram(pass_rates, bins=20, range=(0, 100))

    # Per-video pass rates
    per_video = []
    for vs in video_stats.values():
        vs["pass_rate"] = round(
            100.0 * vs["valid_frames"] / vs["total_frames"]
            if vs["total_frames"] > 0 else 0.0, 1)
        per_video.append(vs)

    aggregate = {
        "total_segments": total_segments,
        "alive_segments": alive,
        "dead_segments": dead,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "per_filter": per_filter,
        "pass_rate_histogram": hist.tolist(),
        "per_video": per_video,
    }
    return aggregate, segment_rows


class CurationOverview(Query):
    name = "Curation Overview"
    description = "Aggregate filter statistics: alive/dead segments, per-filter rejection breakdown"

    def build_params(self):
        pr = st.sidebar.slider("Pass-rate range (%)", 0, 100, (0, 100),
                               key="co_pass_range")
        return {"min_pass": pr[0], "max_pass": pr[1]}

    def execute(self, root, segments, params):
        db_path = params.get("db_path", "")
        if not db_path or not Path(db_path).exists():
            return QueryOutput([], "table", "Curation Overview",
                               "No curation DB found. Set the DB path in the sidebar.")

        mtime = os.path.getmtime(db_path)
        aggregate, segment_rows = _compute_aggregate(db_path, mtime)

        # Filter segment list by pass-rate range
        lo, hi = params["min_pass"], params["max_pass"]
        filtered = [r for r in segment_rows
                    if lo <= r["pass_rate"] <= hi]
        filtered.sort(key=lambda r: r["pass_rate"])

        # First result carries aggregate data
        results = []
        if filtered:
            first = filtered[0].copy()
            first["_aggregate"] = aggregate
            results.append(SegmentResult(first["segment"], metadata=first))
            for r in filtered[1:]:
                results.append(SegmentResult(r["segment"], metadata=r))
        else:
            results.append(SegmentResult("(none)", metadata={"_aggregate": aggregate}))

        return QueryOutput(
            results, "filter_summary", "Curation Overview",
            f"{aggregate['alive_segments']:,} alive / "
            f"{aggregate['dead_segments']:,} dead segments "
            f"({aggregate['total_segments']:,} total)")
