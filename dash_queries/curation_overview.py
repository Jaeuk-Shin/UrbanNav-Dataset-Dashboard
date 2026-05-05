"""Aggregate curation-filter statistics across the full dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from dash_query import Query
from dash_types import QueryOutput, SegmentResult

# Reuse curation internals for filter re-decomposition
from curation.database import get_connection
from curation.filters import FilterConfig, _smooth


# ── Filter decomposition helpers ────────────────────────────────────────


def _decompose_filter_rejections(
    metrics_row: dict[str, bytes],
    num_frames: int,
    cfg: FilterConfig,
) -> dict[str, int]:
    """Re-apply each filter individually and return per-filter rejection counts.

    *metrics_row* maps column names to raw BLOB bytes from segment_filter_data.
    """
    hw = cfg.avg_window
    counts: dict[str, int] = {}

    def _load(key: str) -> np.ndarray:
        blob = metrics_row.get(key)
        if blob is None:
            return np.zeros(num_frames, dtype=np.float32)
        return np.frombuffer(blob, dtype=np.float32)

    vel = _load("velocities")
    nonzero_vel = vel[vel > 1e-8]
    median_vel = float(np.median(nonzero_vel)) if len(nonzero_vel) > 0 else 0.0

    # 1) forward-camera angle
    s = _smooth(_load("forward_camera_angles"), hw)
    counts["forward_camera_angle"] = int(np.sum(s > cfg.forward_camera_max_angle))

    # 2) roll changes
    s = _smooth(_load("roll_changes"), hw)
    counts["roll_change"] = int(np.sum(s > cfg.max_roll_change))

    # 3) pitch changes
    s = _smooth(_load("pitch_changes"), hw)
    counts["pitch_change"] = int(np.sum(s > cfg.max_pitch_change))

    # 4) yaw changes
    s = _smooth(_load("yaw_changes"), hw)
    counts["yaw_change"] = int(np.sum(s > cfg.max_yaw_change))

    # 5) abs pitch
    s = _smooth(_load("abs_pitch"), hw)
    counts["abs_pitch"] = int(np.sum(s > cfg.max_abs_pitch))

    # 6) abs roll
    s = _smooth(_load("abs_roll"), hw)
    counts["abs_roll"] = int(np.sum(s > cfg.max_abs_roll))

    # 7) velocity spikes
    if median_vel > 0:
        spike_thresh = cfg.velocity_spike_factor * median_vel
        counts["velocity_spike"] = int(np.sum(vel > spike_thresh))
    else:
        counts["velocity_spike"] = 0

    # 8) height changes
    s = _smooth(_load("height_changes"), hw)
    counts["height_change"] = int(np.sum(s > cfg.max_height_change_ratio))

    # 9) sustained slow (approximate: count frames in long slow runs)
    if median_vel > 0:
        vel_smooth = _smooth(vel, hw)
        slow_thresh = cfg.sustained_slow_factor * median_vel
        is_slow = vel_smooth < slow_thresh
        changes = np.diff(is_slow.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if is_slow[0]:
            starts = np.concatenate([[0], starts])
        if is_slow[-1]:
            ends = np.concatenate([ends, [num_frames]])
        cnt = 0
        for a, b in zip(starts, ends):
            if b - a >= cfg.sustained_slow_frames:
                cnt += b - a
        counts["sustained_slow"] = cnt
    else:
        counts["sustained_slow"] = 0

    # 10) stop-without-reasons is DB-dependent (annotation queries)
    # We estimate from valid_mask vs sum of other masks
    # (exact count would require re-querying annotations)

    return counts


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

        # Per-filter decomposition
        metrics_row = {
            "velocities": r["velocities"],
            "forward_camera_angles": r["forward_camera_angles"],
            "roll_changes": r["roll_changes"],
            "pitch_changes": r["pitch_changes"],
            "yaw_changes": r["yaw_changes"],
            "abs_pitch": r["abs_pitch"],
            "abs_roll": r["abs_roll"],
            "height_changes": r["height_changes"],
        }
        filt_counts = _decompose_filter_rejections(metrics_row, nf, cfg)
        for k, v in filt_counts.items():
            per_filter[k] = per_filter.get(k, 0) + v

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
