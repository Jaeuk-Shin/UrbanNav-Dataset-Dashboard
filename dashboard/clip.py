"""Clip + trajectory playback for thumbnail visualizers.

The dataset uses per-segment ``.mp4`` files at native (~30 fps) indexing, so
we reuse them directly: ``st.video(path, start_time=s, end_time=e)`` tells
the browser which range to play via HTTP Range requests — no clip encoding.

Alongside the clip we render a 3D trajectory (pose BLOB from the curation
DB) with the clicked frame highlighted.  Two render modes:

* **Triads** (Plotly) — RGB orientation triad at every pose in the window.
* **Textured planes** (Three.js) — N camera quads sampled along the window,
  each textured with the corresponding MP4 frame and oriented by the
  camera quaternion.  Useful for seeing what the camera was looking at.

Pose convention: DPVO/OpenCV — x right, y DOWN, z forward.
``Rot.from_quat([qx,qy,qz,qw]).as_matrix()`` returns a proper right-handed
matrix (det=+1, cross(col0,col1)=col2); columns are the camera basis in
world coords.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial.transform import Rotation as Rot

from curation.database import get_connection

VIDEO_FPS = 30.0
CLIP_DURATION_SEC = 5.0
N_TEXTURED_PLANES = 5
TEXTURE_MAX_WIDTH = 320


def play_button(idx: int, segment: str, frame_id: str,
                namespace: str = "play") -> None:
    """Render a ``▶ Play 5s`` button; selection is stashed in session state."""
    if st.button("▶ Play 5s", key=f"{namespace}_{idx}_{segment}_{frame_id}",
                 use_container_width=True):
        st.session_state["_clip"] = (segment, frame_id)


@st.cache_data(show_spinner=False)
def _video_frame_count(mp4_path: str) -> int | None:
    cap = cv2.VideoCapture(mp4_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def _video_aspect(mp4_path: str) -> float:
    cap = cv2.VideoCapture(mp4_path)
    try:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return float(w) / float(h) if w > 0 and h > 0 else 16.0 / 9.0
    finally:
        cap.release()


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


@st.cache_data(show_spinner=False)
def _frame_jpeg_b64(mp4_path: str, vid_idx: int,
                     max_w: int = TEXTURE_MAX_WIDTH) -> str | None:
    """Extract one MP4 frame, return base64-encoded JPEG (no data URI prefix)."""
    cap = cv2.VideoCapture(mp4_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(vid_idx)))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    finally:
        cap.release()


def _axis_segments(origins: np.ndarray, ends: np.ndarray
                    ) -> tuple[list, list, list]:
    """Interleave [origin, end, None] for a single Plotly line trace."""
    xs, ys, zs = [], [], []
    for o, e in zip(origins, ends):
        xs += [float(o[0]), float(e[0]), None]
        ys += [float(o[1]), float(e[1]), None]
        zs += [float(o[2]), float(e[2]), None]
    return xs, ys, zs


def _render_triads(pose: np.ndarray, pose_idx: int,
                    window_half: int = 12,
                    height: int = 420) -> None:
    """3D trajectory + RGB orientation triads (Plotly).

    Triads are right-handed (verified: det=1, x×y=z).  Camera up is set to
    world -y so DPVO's y-down convention displays with "up" pointing up.
    """
    N = len(pose)
    pose_idx = max(0, min(pose_idx, N - 1))
    lo = max(0, pose_idx - window_half)
    hi = min(N, pose_idx + window_half + 1)
    sub = pose[lo:hi]
    origins = sub[:, :3]
    rot = Rot.from_quat(sub[:, 3:]).as_matrix()  # (W, 3, 3)

    span = float(max(
        origins[:, 0].ptp(), origins[:, 1].ptp(), origins[:, 2].ptp(), 0.1
    ))
    axis_len = max(span * 0.08, 0.05)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
        mode="lines+markers",
        line=dict(color="#999", width=2),
        marker=dict(size=2, color="#555"),
        hoverinfo="skip", name="path",
    ))

    for axis_idx, color, name in (
        (0, "red", "x (right)"),
        (1, "green", "y (down)"),
        (2, "blue", "z (fwd)"),
    ):
        ends = origins + axis_len * rot[:, :, axis_idx]
        xs, ys, zs = _axis_segments(origins, ends)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=4),
            hoverinfo="skip", name=name,
        ))

    cx, cy, cz = pose[pose_idx, :3]
    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[cz], mode="markers",
        marker=dict(color="orange", size=7, symbol="diamond"),
        name=f"frame (pose #{pose_idx})",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y (down)", zaxis_title="z (fwd)",
            aspectmode="data",
            camera=dict(up=dict(x=0, y=-1, z=0),
                         eye=dict(x=1.5, y=-1.2, z=1.5)),
        ),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1,
                     xanchor="center", x=0.5),
    )
    # Default theme="streamlit" lets the chart inherit the runtime theme.
    st.plotly_chart(fig, use_container_width=True)


_THREEJS_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
  #c { width:100%; height:__HEIGHT__px; display:block; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script type="importmap">
{"imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const data = __DATA__;
const canvas = document.getElementById('c');

function size() {
  const r = canvas.getBoundingClientRect();
  return [Math.max(r.width|0, 100), Math.max(r.height|0, 100)];
}
let [W0, H0] = size();

// Detect the rendered Streamlit theme at runtime: try the parent
// document's body bg first (works for both config-set and Settings-menu
// themes since the component iframe is same-origin), then fall back to
// the OS prefers-color-scheme, then plain white.
function detectBg() {
  try {
    const c = window.parent.getComputedStyle(window.parent.document.body)
                          .backgroundColor;
    if (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') return c;
  } catch (e) {}
  if (window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return '#0E1117';
  }
  return '#FFFFFF';
}
const bg = detectBg();
document.body.style.background = bg;

const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W0, H0, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(new THREE.Color(bg), 1.0);

const scene = new THREE.Scene();

const path = data.path;
const cur  = data.current;
const cx = path.reduce((s,p)=>s+p[0],0)/path.length;
const cy = path.reduce((s,p)=>s+p[1],0)/path.length;
const cz = path.reduce((s,p)=>s+p[2],0)/path.length;

let span = 0.5;
for (const p of path) span = Math.max(span, Math.hypot(p[0]-cx, p[1]-cy, p[2]-cz));
const planeH = span * 0.30;
const planeW = planeH * data.aspect;

const camera = new THREE.PerspectiveCamera(45, W0/H0, 0.01, span*100);
// DPVO has y DOWN, so set scene-camera up to (0,-1,0): world "up" appears
// up on screen.  Position: oblique view from above-front-right.
camera.position.set(cx + span*2.2, cy - span*1.8, cz + span*2.2);
camera.up.set(0, -1, 0);

const controls = new OrbitControls(camera, canvas);
controls.target.set(cx, cy, cz);
controls.update();

// path line
{
  const g = new THREE.BufferGeometry();
  const v = new Float32Array(path.length*3);
  path.forEach((p,i)=>{ v[3*i]=p[0]; v[3*i+1]=p[1]; v[3*i+2]=p[2]; });
  g.setAttribute('position', new THREE.BufferAttribute(v,3));
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({color:0x666666})));
}

// current pose marker
{
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(planeH*0.06, 16, 16),
    new THREE.MeshBasicMaterial({color:0xff8800}));
  m.position.set(cur[0], cur[1], cur[2]);
  scene.add(m);
}

// Per-pose RGB triad: red=x(right), green=y(down), blue=z(forward).
function addTriad(p, q, len) {
  const grp = new THREE.Group();
  grp.position.set(p[0], p[1], p[2]);
  grp.quaternion.set(q[0], q[1], q[2], q[3]); // THREE expects xyzw — same as ours
  const cols = [0xff3333, 0x33cc33, 0x3366ff];
  for (let i = 0; i < 3; i++) {
    const a = new THREE.Vector3(0,0,0);
    const b = new THREE.Vector3(0,0,0); b.setComponent(i, len);
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    grp.add(new THREE.Line(g, new THREE.LineBasicMaterial({color: cols[i]})));
  }
  scene.add(grp);
}

// Textured camera planes — one per sampled pose.
// PlaneGeometry default lies in local x-y with normal +z; after applying the
// camera quaternion the plane sits in the camera image plane with normal
// along camera +z (forward).  We pre-rotate 180° about local x so that the
// plane's UV "top" maps to camera -y (= world up under DPVO y-down),
// keeping the texture right-side-up under the scene-camera up=(0,-1,0).
const loader = new THREE.TextureLoader();
for (const pl of data.planes) {
  loader.load('data:image/jpeg;base64,'+pl.tex, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    const geom = new THREE.PlaneGeometry(planeW, planeH);
    geom.rotateX(Math.PI);
    const mat = new THREE.MeshBasicMaterial({
      map: tex, side: THREE.DoubleSide, transparent: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(pl.p[0], pl.p[1], pl.p[2]);
    mesh.quaternion.set(pl.q[0], pl.q[1], pl.q[2], pl.q[3]);
    scene.add(mesh);
    addTriad(pl.p, pl.q, planeH * 0.55);
  });
}

function animate() {
  const [w, h] = size();
  if (w !== W0 || h !== H0) {
    W0 = w; H0 = h;
    renderer.setSize(W0, H0, false);
    camera.aspect = W0/H0;
    camera.updateProjectionMatrix();
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
</script>
</body>
</html>
"""


def _render_textured_planes(pose: np.ndarray, pose_idx: int,
                             window_half: int, height: int,
                             mp4_path: str, n_video: int | None,
                             n_planes: int = N_TEXTURED_PLANES) -> None:
    """Three.js scene: trajectory + ``n_planes`` textured camera quads."""
    N = len(pose)
    pose_idx = max(0, min(pose_idx, N - 1))
    lo = max(0, pose_idx - window_half)
    hi = min(N, pose_idx + window_half + 1)
    sub = pose[lo:hi]
    W = len(sub)
    if W == 0:
        st.caption("(Empty pose window)")
        return

    n_planes = min(n_planes, W)
    sample_local = np.linspace(0, W - 1, n_planes, dtype=int)
    aspect = _video_aspect(mp4_path)

    planes = []
    for li in sample_local:
        gi = lo + int(li)
        if n_video and n_video > 0:
            vid_idx = int(round(gi * n_video / N))
        else:
            vid_idx = gi * 6
        b64 = _frame_jpeg_b64(mp4_path, vid_idx)
        if b64 is None:
            continue
        x, y, z, qx, qy, qz, qw = sub[li]
        planes.append({
            "p": [float(x), float(y), float(z)],
            "q": [float(qx), float(qy), float(qz), float(qw)],
            "tex": b64,
        })

    payload = json.dumps({
        "path": [[float(v) for v in p] for p in sub[:, :3]],
        "current": [float(v) for v in pose[pose_idx, :3]],
        "planes": planes,
        "aspect": aspect,
    })

    html = (_THREEJS_TEMPLATE
            .replace("__DATA__", payload)
            .replace("__HEIGHT__", str(height)))
    components.html(html, height=height + 10)


def _video_frame_to_pose_idx(vid_frame: int, n_pose: int,
                             n_video: int | None) -> int:
    """Map video frame index (native fps) to pose row index."""
    if n_video and n_video > 0:
        return min(int(vid_frame * n_pose / n_video), n_pose - 1)
    return min(vid_frame // 6, n_pose - 1)


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
