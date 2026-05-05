"""Per-segment filter diagnostic: metric timelines with threshold overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from dash_query import Query
from dash_types import QueryOutput, SegmentResult

from curation.database import get_connection
from curation.filters import FilterConfig, _smooth


def _build_individual_masks(
    metrics: dict[str, list[float]],
    num_frames: int,
    cfg: FilterConfig,
) -> dict[str, list[bool]]:
    """Re-apply each filter to produce individual boolean masks."""
    hw = cfg.avg_window
    masks: dict[str, list[bool]] = {}

    def _arr(key: str) -> np.ndarray:
        return np.array(metrics[key], dtype=np.float32)

    vel = _arr("velocities")
    nonzero_vel = vel[vel > 1e-8]
    median_vel = float(np.median(nonzero_vel)) if len(nonzero_vel) > 0 else 0.0

    # 1) forward-camera angle
    s = _smooth(_arr("forward_camera_angles"), hw)
    masks["forward_camera_angle"] = (s <= cfg.forward_camera_max_angle).tolist()

    # 2) roll changes
    s = _smooth(_arr("roll_changes"), hw)
    masks["roll_change"] = (s <= cfg.max_roll_change).tolist()

    # 3) pitch changes
    s = _smooth(_arr("pitch_changes"), hw)
    masks["pitch_change"] = (s <= cfg.max_pitch_change).tolist()

    # 4) yaw changes
    s = _smooth(_arr("yaw_changes"), hw)
    masks["yaw_change"] = (s <= cfg.max_yaw_change).tolist()

    # 5) abs pitch
    s = _smooth(_arr("abs_pitch"), hw)
    masks["abs_pitch"] = (s <= cfg.max_abs_pitch).tolist()

    # 6) abs roll
    s = _smooth(_arr("abs_roll"), hw)
    masks["abs_roll"] = (s <= cfg.max_abs_roll).tolist()

    # 7) velocity spikes
    if median_vel > 0:
        spike_thresh = cfg.velocity_spike_factor * median_vel
        masks["velocity_spike"] = (vel <= spike_thresh).tolist()
    else:
        masks["velocity_spike"] = [True] * num_frames

    # 8) height changes
    s = _smooth(_arr("height_changes"), hw)
    masks["height_change"] = (s <= cfg.max_height_change_ratio).tolist()

    # 9) sustained slow
    slow_mask = np.ones(num_frames, dtype=bool)
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
        for a, b in zip(starts, ends):
            if b - a >= cfg.sustained_slow_frames:
                slow_mask[a:b] = False
    masks["sustained_slow"] = slow_mask.tolist()

    # 10) stop-without-reasons (from DB valid_mask minus other masks)
    # Exact decomposition requires annotation data; we approximate by
    # comparing the DB valid_mask against the AND of filters 1-9.
    # Frames that pass 1-9 but fail the combined mask were killed by #10.

    return masks


class FilterDiagnostic(Query):
    name = "Filter Diagnostic"
    description = "Per-segment deep dive: metric timelines with filter thresholds"

    def build_params(self):
        seg = st.sidebar.text_input("Segment name", "", key="fd_segment")
        show_thresh = st.sidebar.checkbox("Show thresholds", True,
                                          key="fd_thresholds")
        return {"segment": seg, "show_thresholds": show_thresh}

    def execute(self, root, segments, params):
        db_path = params.get("db_path", "")
        seg_name = params.get("segment", "").strip()

        if not db_path or not Path(db_path).exists():
            return QueryOutput([], "table", "Filter Diagnostic",
                               "No curation DB found.")
        if not seg_name:
            return QueryOutput([], "table", "Filter Diagnostic",
                               "Enter a segment name in the sidebar.")

        conn = get_connection(db_path, readonly=True)

        # Look up segment
        row = conn.execute(
            """SELECT s.segment_id, s.num_frames,
                      f.velocities, f.forward_camera_angles,
                      f.roll_changes, f.pitch_changes, f.yaw_changes,
                      f.abs_pitch, f.abs_roll, f.height_changes,
                      f.valid_mask
               FROM segments s
               JOIN segment_filter_data f ON s.segment_id = f.segment_id
               WHERE s.name = ?""",
            (seg_name,),
        ).fetchone()
        conn.close()

        if row is None:
            return QueryOutput([], "table", "Filter Diagnostic",
                               f"Segment '{seg_name}' not found or not filtered.")

        nf = row["num_frames"]
        cfg = FilterConfig()

        # Decode metric BLOBs
        metric_keys = [
            "velocities", "forward_camera_angles",
            "roll_changes", "pitch_changes", "yaw_changes",
            "abs_pitch", "abs_roll", "height_changes",
        ]
        metrics: dict[str, list[float]] = {}
        for k in metric_keys:
            blob = row[k]
            if blob:
                arr = np.frombuffer(blob, dtype=np.float32)
                metrics[k] = arr.tolist()
            else:
                metrics[k] = [0.0] * nf

        # Decode combined valid mask
        combined = np.frombuffer(
            row["valid_mask"], dtype=np.uint8
        ).astype(bool)
        pass_rate = 100.0 * combined.sum() / nf if nf > 0 else 0.0

        # Build individual filter masks
        individual_masks = _build_individual_masks(metrics, nf, cfg)

        # Infer stop-without-reasons mask from combined vs AND of 1-9
        and_of_others = np.ones(nf, dtype=bool)
        for m in individual_masks.values():
            and_of_others &= np.array(m, dtype=bool)
        # Frames that pass 1-9 but fail combined were killed by filter #10
        stop_mask = ~(and_of_others & ~combined)
        individual_masks["stop_without_reasons"] = stop_mask.tolist()
        individual_masks["combined"] = combined.tolist()

        # Velocity spike threshold is dynamic (median-based)
        vel = np.array(metrics["velocities"], dtype=np.float32)
        nonzero_vel = vel[vel > 1e-8]
        median_vel = float(np.median(nonzero_vel)) if len(nonzero_vel) > 0 else 0.0

        thresholds: dict[str, float] = {}
        if params.get("show_thresholds", True):
            thresholds = {
                "forward_camera_angle": cfg.forward_camera_max_angle,
                "roll_change": cfg.max_roll_change,
                "pitch_change": cfg.max_pitch_change,
                "yaw_change": cfg.max_yaw_change,
                "abs_pitch": cfg.max_abs_pitch,
                "abs_roll": cfg.max_abs_roll,
                "velocity_spike": cfg.velocity_spike_factor * median_vel
                                  if median_vel > 0 else None,
                "height_change": cfg.max_height_change_ratio,
            }
            thresholds = {k: v for k, v in thresholds.items() if v is not None}

        md = {
            "segment": seg_name,
            "num_frames": nf,
            "pass_rate": round(pass_rate, 1),
            "metrics": metrics,
            "masks": individual_masks,
            "thresholds": thresholds,
        }
        result = SegmentResult(seg_name, score=pass_rate, metadata=md)
        return QueryOutput(
            [result], "filter_timeline", "Filter Diagnostic",
            f"{seg_name}: {nf} frames, {pass_rate:.1f}% valid")
