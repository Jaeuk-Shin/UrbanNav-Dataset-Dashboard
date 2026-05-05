"""Inline clip + trajectory player rendered atop visualizer result lists.

The dataset uses per-segment ``.mp4`` files at native (~30 fps) indexing, so
we reuse them directly: ``st.video(path, start_time=s, end_time=e)`` tells
the browser which range to play via HTTP Range requests — no clip encoding.

Alongside the clip we render a 3D trajectory (pose BLOB from the curation
DB) with the clicked frame highlighted.  Two render modes:

* **Triads** (Plotly) — RGB orientation triad at every pose in the window.
* **Textured planes** (Three.js) — N camera quads sampled along the window,
  each textured with the corresponding MP4 frame and oriented by the
  camera quaternion.  Useful for seeing what the camera was looking at.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from curation.database import get_connection

from ._video_io import (
    VIDEO_FPS,
    _video_frame_count,
    _video_frame_to_pose_idx,
)
from .clip_trajectory import _render_textured_planes, _render_triads

CLIP_DURATION_SEC = 5.0


def play_button(idx: int, segment: str, frame_id: str,
                namespace: str = "play") -> None:
    """Render a ``▶ Play 5s`` button; selection is stashed in session state."""
    if st.button("▶ Play 5s", key=f"{namespace}_{idx}_{segment}_{frame_id}",
                 use_container_width=True):
        st.session_state["_clip"] = (segment, frame_id)


@st.cache_data(show_spinner=False)
def _load_pose(db_path: str, seg: str, _mtime: float) -> np.ndarray | None:
    """Load (N, 7) pose (x y z qx qy qz qw) from segment_poses for *seg*."""
    conn = get_connection(db_path, readonly=True)
    row = conn.execute(
        """SELECT sp.pose_data FROM segment_poses sp
           JOIN segments s ON sp.segment_id = s.segment_id
           WHERE s.name = ?""",
        (seg,),
    ).fetchone()
    conn.close()
    if row is None or row["pose_data"] is None:
        return None
    return np.frombuffer(row["pose_data"], dtype=np.float64).reshape(-1, 7)


def show_selected_clip(root: Path,
                       duration: float = CLIP_DURATION_SEC) -> None:
    """Render the clicked clip + trajectory at the top of the results view."""
    clip = st.session_state.get("_clip")
    if not clip:
        return
    seg, fid = clip

    vid = root / "rgb" / f"{seg}.mp4"
    if not vid.exists():
        st.warning(f"No MP4 found for segment: {seg}")
        return
    try:
        fi = int(fid)
    except ValueError:
        return

    center = fi / VIDEO_FPS
    start = max(0.0, center - duration / 2)
    end = center + duration / 2

    st.markdown(f"**Clip** — `{seg}` @ frame {fi} ({center:.1f}s)")

    vid_col, traj_col = st.columns([1, 1])
    with vid_col:
        st.video(str(vid), start_time=int(start), end_time=int(end),
                 autoplay=True, muted=True)

    with traj_col:
        mode = st.selectbox(
            "Trajectory view",
            ["Triads (Plotly)", "Textured planes (Three.js)"],
            key="_clip_traj_mode",
            index=0,
        )
        db_path = st.session_state.get("_clip_db_path", "")
        if db_path and Path(db_path).exists():
            pose = _load_pose(db_path, seg, os.path.getmtime(db_path))
            if pose is not None and len(pose) > 0:
                n_video = _video_frame_count(str(vid))
                pose_idx = _video_frame_to_pose_idx(fi, len(pose), n_video)
                if n_video and n_video > 0:
                    ratio = len(pose) / n_video
                else:
                    ratio = 1.0 / 6.0
                window_half = max(3, int(round(duration * 0.5
                                                * VIDEO_FPS * ratio)))
                if mode.startswith("Triads"):
                    _render_triads(pose, pose_idx,
                                    window_half=window_half, height=420)
                else:
                    _render_textured_planes(
                        pose, pose_idx,
                        window_half=window_half, height=420,
                        mp4_path=str(vid), n_video=n_video,
                    )
            else:
                st.caption("(No pose data in DB for this segment)")
        else:
            st.caption("(Set the Curation DB path in the sidebar "
                       "to see trajectories)")

    if st.button("✕ Close clip", key="_clip_close"):
        del st.session_state["_clip"]
        st.rerun()
    st.divider()
