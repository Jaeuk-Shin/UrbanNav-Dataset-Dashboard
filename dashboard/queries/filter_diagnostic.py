"""Per-segment filter diagnostic: metric timelines with threshold overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from ..query import Query
from ..types import QueryOutput, SegmentResult

from curation.database import get_connection
from curation.filters import FilterConfig, compute_filter_masks


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
        metrics: dict[str, np.ndarray] = {}
        for k in metric_keys:
            blob = row[k]
            if blob:
                metrics[k] = np.frombuffer(blob, dtype=np.float32)
            else:
                metrics[k] = np.zeros(nf, dtype=np.float32)

        # Decode combined valid mask
        combined = np.frombuffer(
            row["valid_mask"], dtype=np.uint8
        ).astype(bool)
        pass_rate = 100.0 * combined.sum() / nf if nf > 0 else 0.0

        # Per-filter masks (filters 1-9 + sustained_slow; no DB conn → no #10)
        nd_masks = compute_filter_masks(metrics, cfg)

        # Infer stop-without-reasons mask from combined vs AND of 1-9
        and_of_others = np.logical_and.reduce(list(nd_masks.values()))
        # Frames that pass 1-9 but fail combined were killed by filter #10
        stop_mask = ~(and_of_others & ~combined)

        individual_masks: dict[str, list[bool]] = {
            k: m.tolist() for k, m in nd_masks.items()
        }
        individual_masks["stop_without_reasons"] = stop_mask.tolist()
        individual_masks["combined"] = combined.tolist()

        # Metrics for the visualizer must be plain lists (truthiness check).
        metrics_list = {k: m.tolist() for k, m in metrics.items()}

        # Velocity spike threshold is dynamic (median-based)
        vel = metrics["velocities"]
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
            "metrics": metrics_list,
            "masks": individual_masks,
            "thresholds": thresholds,
        }
        result = SegmentResult(seg_name, score=pass_rate, metadata=md)
        return QueryOutput(
            [result], "filter_timeline", "Filter Diagnostic",
            f"{seg_name}: {nf} frames, {pass_rate:.1f}% valid")
