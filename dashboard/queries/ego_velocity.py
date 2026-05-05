"""Ego velocity query."""

import numpy as np
import streamlit as st

from ..query import Query
from ..types import QueryOutput, SegmentResult
from ..loaders import load_poses, DATASET_FPS


class EgoVelocity(Query):
    name = "Ego Velocity"
    description = "Segments whose average XZ-plane velocity exceeds a threshold"

    def build_params(self):
        return {
            "min_vel": st.sidebar.slider(
                "Min avg velocity (m/s)", 0.0, 3.0, 0.5, 0.05, key="v_min"
            ),
            "fps": st.sidebar.number_input(
                "Dataset FPS (approx)", 0.5, 30.0, DATASET_FPS, 0.5, key="v_fps"
            ),
        }

    def execute(self, root, segments, params):
        fps = params["fps"]
        out = []
        for seg in segments:
            pf = root / "pose" / f"{seg}.txt"
            if not pf.exists():
                continue
            p = load_poses(str(pf))
            if len(p) < 2:
                continue
            xz = p[:, [2, 4]]
            disp = np.linalg.norm(np.diff(xz, axis=0), axis=1)
            vel = disp * fps
            avg_v = float(np.mean(vel))
            if avg_v >= params["min_vel"]:
                out.append(SegmentResult(seg, avg_v, {
                    "segment": seg,
                    "avg_velocity": round(avg_v, 3),
                    "max_velocity": round(float(np.max(vel)), 3),
                    "positions": p[:, 2:5].tolist(),
                    "velocities": vel.tolist(),
                }))
        out.sort(key=lambda x: -x.score)
        return QueryOutput(
            out, "trajectory",
            f"Segments with avg velocity >= {params['min_vel']} m/s",
            f"{len(out)} segments",
        )
