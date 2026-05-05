# Quality Filters

The curation pipeline applies ten per-frame filters to YouTube walking videos.
Each filter produces a boolean mask (True = keep); the combined mask is the
logical AND. Per-frame metrics are smoothed by a uniform sliding-window
average (`avg_window`, default ±5 frames) before thresholding to reduce noise
sensitivity.

All thresholds live on `FilterConfig` in `curation/filters.py` and are
overridable via `python -m curation filter` flags. Defaults are intentionally
permissive — tighten per experiment.

## Pose convention

Poses come from deep patch visual odometry (DPVO) and follow the OpenCV camera
convention: x right, **y down**, z forward. The world frame is anchored to
the first camera pose, so absolute pitch/roll are zero at frame 0.

The pose file (`pose/*.txt`) has eight columns
`frame_index x y z qx qy qz qw`. Quaternions are xyzw.

---

## 1. Forward-vs-camera angle (`forward_camera_max_angle`, default 60°)

YouTube videos are filmed by humans whose camera is not rigidly attached to
their body, so the velocity vector and the camera's forward direction can
diverge (the walker turns their head, holds the phone sideways, etc.).

For each frame, compute the angle between the per-frame translation vector
and the camera's z-axis (third column of the rotation matrix). The result is
in [0°, 180°]: 0° = perfectly forward, 180° = backward. Frames with near-zero
velocity get angle 0 (no meaningful direction).

## 2. Roll changes (`max_roll_change`, default 30°/frame)

Per-frame absolute change in roll, derived from the ZYX Euler decomposition.
Catches sudden tilts (the camera rotating around its forward axis).

## 3. Pitch changes (`max_pitch_change`, default 20°/frame)

Per-frame absolute change in pitch. Pitch is extracted directly via
`arcsin(R[2, 0])` (rather than from the Euler decomposition) to stay robust
near gimbal lock at ±90°. Catches sudden up/down tilts (looking at the sky,
the ground).

## 4. Yaw changes (`max_yaw_change`, default 45°/frame)

Per-frame absolute change in yaw, with ±180° wrap-around handling. Catches
fast turning that the wheeled-robot policy is unlikely to imitate.

## 5. Absolute pitch (`max_abs_pitch`, default 30°)

Sustained tilt up or down. Cumulative pitch from the world frame, taken via
the same gimbal-safe `arcsin` extraction.

## 6. Absolute roll (`max_abs_roll`, default 20°)

Sustained sideways tilt (phone held at an angle).

## 7. Velocity spikes (`velocity_spike_factor`, default 10× segment median)

Per-frame Euclidean distance between consecutive positions. Frames whose
velocity exceeds `factor × median(non-zero velocities)` of the same segment
are dropped. The threshold is scale-invariant by construction — DPVO scale
varies between videos.

This catches DPVO tracking failures (pose teleports several metres in one
frame).

## 8. Height changes (`max_height_change_ratio`, default 0.3)

Per-frame `|Δy| / |Δxyz|` (the y-axis is vertical in DPVO's camera frame).
A wheeled robot on flat ground stays near zero. High values flag stairs,
escalators, steep slopes, and VO drift.

## 9. Sustained low speed (`sustained_slow_frames`/`sustained_slow_factor`)

A frame is *slow* when its smoothed velocity is below
`sustained_slow_factor × median_velocity` (default 0.15×). When the segment
contains a run of ≥`sustained_slow_frames` (default 20) consecutive slow
frames, every frame in that run is rejected.

This catches long pauses regardless of cause — e.g. the walker stops to look
around, talk, or wait at a long signal — without depending on annotations.

## 10. Stop-without-reasons (annotation-aware)

Stops are only valid when the scene justifies them: a crosswalk in front of
the walker, or one or more nearby pedestrians. The check runs only when the
DB contains annotations for the segment.

For each frame whose smoothed velocity is below `stop_velocity_threshold`
(default 0.005), find the nearest annotated frame in `frame_annotations`.
The frame is *justified* when **either**:

- a `crosswalk` detection at that annotated frame has confidence ≥
  `crosswalk_confidence` (default 0.3), **or**
- at least `min_pedestrians` (default 1) `pedestrian` detections have
  confidence ≥ `pedestrian_confidence` (default 0.4).

Annotated frames farther than `stop_max_ann_distance` pose frames (default
10, one annotation interval at 0.2 fps with `DATASET_FPS=2.0`) from the
stopped frame don't count. Frames whose nearest annotation is too far away,
or is unjustified, are rejected.

---

## Where filter results live

After `python -m curation filter`, the per-frame metrics and the combined
`valid_mask` are stored as numpy BLOBs in `segment_filter_data`. Decode with
`np.frombuffer(blob, dtype=np.float32)` (or `np.uint8` for `valid_mask`).

The dashboard's "Curation Overview" and "Filter Diagnostic" queries inspect
these BLOBs directly. `python -m curation build-lut` consumes the
`valid_mask` to emit `(segment_idx, pose_start)` training windows.
