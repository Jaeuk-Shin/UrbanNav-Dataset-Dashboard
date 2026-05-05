"""Data quality filters for the curation pipeline.

Filters operate on pose trajectories and (optionally) detection annotations.
Each filter produces a per-frame boolean mask (True = valid).  The combined
mask is the intersection (logical AND) of all individual masks.

All angular metrics are averaged over a sliding window before thresholding
to reduce noise sensitivity (as recommended in docs/FILTERS.md).

Usage::

    python -m curation.cli filter --db dataset.db [filter options]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .database import get_connection
from .poses import load_pose_from_blob, load_pose_from_text, smooth_window

# ------------------------------------------------------------------ #
#  Filter configuration                                               #
# ------------------------------------------------------------------ #


@dataclass
class FilterConfig:
    """Thresholds and parameters for all filters.

    Defaults are intentionally permissive; tighten per-experiment.
    """

    # --- Forward-vs-camera vector ---
    # Maximum angle (degrees) between velocity direction and camera z-axis.
    # Range is [0, 180]: 0 = perfectly forward, 180 = backward.
    forward_camera_max_angle: float = 60.0

    # --- Orientation changes (per-frame) ---
    # Maximum per-frame roll change (degrees).
    max_roll_change: float = 30.0
    # Maximum per-frame pitch change (degrees).
    max_pitch_change: float = 20.0
    # Maximum per-frame yaw change (degrees).
    max_yaw_change: float = 45.0

    # --- Absolute orientation bounds ---
    # Maximum absolute pitch from horizontal (degrees).  Catches sustained
    # camera tilt up/down (sky-gazing, ground-staring).
    max_abs_pitch: float = 30.0
    # Maximum absolute roll from level (degrees).  Catches sustained sideways
    # tilt (phone held at an angle).
    max_abs_roll: float = 20.0

    # --- Velocity spikes (DPVO tracking failures) ---
    # A frame is flagged when its velocity exceeds this multiple of the
    # segment's median non-zero velocity.  Scale-invariant.
    velocity_spike_factor: float = 10.0

    # --- Height changes (non-flat terrain) ---
    # Maximum ratio |Δy| / |Δxyz| per frame.  For a wheeled robot on flat
    # ground this should be near zero.  Catches stairs, escalators, VO drift.
    # Uses the y-axis as vertical (DPVO camera convention: y is down).
    max_height_change_ratio: float = 0.3

    # --- Sustained low speed ---
    # Minimum consecutive slow frames to trigger filtering.
    sustained_slow_frames: int = 20
    # A frame is "slow" when its velocity is below this fraction of the
    # segment's median non-zero velocity.
    sustained_slow_factor: float = 0.15

    # --- Stop-without-reasons ---
    # Velocity below this threshold (in distance-per-frame) triggers the stop
    # check.  Units depend on the DPVO scale; the default is deliberately low.
    stop_velocity_threshold: float = 0.005
    # Minimum confidence for a pedestrian detection to count.
    pedestrian_confidence: float = 0.4
    # Minimum number of confident pedestrians to justify a stop.
    min_pedestrians: int = 1
    # Crosswalk confidence threshold.
    crosswalk_confidence: float = 0.3
    # Maximum distance (in pose frames) to the nearest annotated frame for a
    # stop justification to be considered relevant.  At DATASET_FPS=2.0 and
    # annotation rate of 0.2 fps (every 5 s), annotations are every 10 pose
    # frames — so the default of 10 means "within one annotation interval".
    stop_max_ann_distance: int = 10

    # --- Shared ---
    # Sliding window half-size (frames) for metric averaging.
    avg_window: int = 5

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "FilterConfig":
        return cls(**json.loads(s))


# ------------------------------------------------------------------ #
#  Individual filters                                                 #
# ------------------------------------------------------------------ #


def compute_velocities(pose: np.ndarray) -> np.ndarray:
    """Compute per-frame velocity magnitude (Euclidean distance per step).

    Returns (N,) array; first frame copies the second frame's value.
    """
    positions = pose[:, :3]  # (N, 3)
    diffs = np.diff(positions, axis=0)  # (N-1, 3)
    speeds = np.linalg.norm(diffs, axis=1)  # (N-1,)
    return np.concatenate([[speeds[0]], speeds])


def compute_forward_camera_angles(
    pose: np.ndarray,
    rot_matrices: np.ndarray | None = None,
) -> np.ndarray:
    """Angle (degrees) between velocity direction and camera forward (z-axis).

    The camera forward is the z-column of the rotation matrix derived from the
    quaternion.  The velocity direction is the normalised translation vector
    between consecutive frames.

    If *rot_matrices* is provided (N, 3, 3), reuses them instead of
    recomputing from quaternions.

    Returns (N,) array in [0, 180].  0 = perfectly forward, 180 = backward.
    Frames with near-zero velocity get angle = 0 (no meaningful direction).
    """
    N = pose.shape[0]
    if N < 2:
        return np.zeros(N, dtype=np.float64)

    positions = pose[:, :3]

    if rot_matrices is None:
        quats = pose[:, 3:]  # (N, 4) xyzw
        rot_matrices = R.from_quat(quats).as_matrix()

    # Camera forward = z-axis of rotation matrix (third column)
    cam_forward = rot_matrices[:, :, 2]  # (N, 3)

    # Velocity vectors
    diffs = np.diff(positions, axis=0)  # (N-1, 3)
    norms = np.linalg.norm(diffs, axis=1)  # (N-1,)
    norms_safe = np.where(norms < 1e-8, 1.0, norms)
    vel_dirs = diffs / norms_safe[:, None]  # (N-1, 3)

    # Vectorised dot product between camera forward and velocity direction
    dots = np.sum(cam_forward[:-1] * vel_dirs, axis=1)  # (N-1,)
    dots = np.clip(dots, -1.0, 1.0)
    raw_angles = np.degrees(np.arccos(dots))  # [0, 180]

    # Zero out angles for stationary frames (no meaningful direction)
    raw_angles[norms < 1e-8] = 0.0

    angles = np.zeros(N, dtype=np.float64)
    angles[:-1] = raw_angles
    angles[-1] = angles[-2]
    return angles


def compute_orientation_changes(
    pose: np.ndarray,
    rot_matrices: np.ndarray | None = None,
    euler: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame roll, pitch, and yaw changes (degrees).

    Returns (roll_changes, pitch_changes, yaw_changes), each shape (N,).
    First frame = 0.

    If *rot_matrices* (N,3,3) and/or *euler* (N,3) are provided, reuses them.

    Pitch is extracted directly via arcsin to avoid gimbal-lock artefacts in
    the Euler decomposition of roll and yaw near ±90° pitch.
    """
    N = pose.shape[0]

    if rot_matrices is None or euler is None:
        quats = pose[:, 3:]  # xyzw
        rotations = R.from_quat(quats)
        if rot_matrices is None:
            rot_matrices = rotations.as_matrix()
        if euler is None:
            euler = rotations.as_euler("ZYX", degrees=True)

    # --- Pitch: extracted from rotation matrix for robustness ---
    pitch = np.degrees(-np.arcsin(np.clip(rot_matrices[:, 2, 0], -1.0, 1.0)))

    # --- Roll and yaw: Euler decomposition ---
    yaw = euler[:, 0]
    roll = euler[:, 2]

    roll_diff = np.abs(np.diff(roll))
    pitch_diff = np.abs(np.diff(pitch))
    yaw_diff = np.abs(np.diff(yaw))
    # Wrap yaw differences (handle +-180 boundary)
    yaw_diff = np.minimum(yaw_diff, 360.0 - yaw_diff)

    roll_changes = np.concatenate([[0.0], roll_diff])
    pitch_changes = np.concatenate([[0.0], pitch_diff])
    yaw_changes = np.concatenate([[0.0], yaw_diff])
    return roll_changes, pitch_changes, yaw_changes


def compute_absolute_orientation(
    pose: np.ndarray,
    rot_matrices: np.ndarray | None = None,
    euler: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute pitch and roll angles (degrees) relative to the initial frame.

    DPVO defines the world frame from the first camera pose, so pitch ≈ 0 and
    roll ≈ 0 when the camera is in its initial (approximately horizontal)
    orientation.

    If *rot_matrices* (N,3,3) and/or *euler* (N,3) are provided, reuses them.

    Returns (abs_pitch, abs_roll), each shape (N,) in non-negative degrees.
    """
    if rot_matrices is None or euler is None:
        quats = pose[:, 3:]
        rotations = R.from_quat(quats)
        if rot_matrices is None:
            rot_matrices = rotations.as_matrix()
        if euler is None:
            euler = rotations.as_euler("ZYX", degrees=True)

    # Pitch from rotation matrix (gimbal-lock safe)
    pitch = np.degrees(-np.arcsin(np.clip(rot_matrices[:, 2, 0], -1.0, 1.0)))
    abs_pitch = np.abs(pitch)

    # Roll from Euler decomposition
    abs_roll = np.abs(euler[:, 2])

    return abs_pitch.astype(np.float64), abs_roll.astype(np.float64)


def compute_height_changes(pose: np.ndarray) -> np.ndarray:
    """Per-frame vertical displacement fraction ``|Δy| / |Δxyz|``.

    Uses the y-axis as vertical (DPVO camera convention: +y is down).
    A wheeled robot on flat ground should have this ratio near zero.
    High values indicate stairs, escalators, steep slopes, or VO drift.

    Returns (N,) array in [0, 1].  First frame copies second frame's value.
    Frames with near-zero total displacement get ratio = 0.
    """
    positions = pose[:, :3]
    diffs = np.diff(positions, axis=0)  # (N-1, 3)
    dy = np.abs(diffs[:, 1])  # y-axis = vertical
    total = np.linalg.norm(diffs, axis=1)
    ratio = np.where(total < 1e-8, 0.0, dy / total)
    return np.concatenate([[ratio[0]] if len(ratio) > 0 else [0.0], ratio])


# ------------------------------------------------------------------ #
#  Annotation helpers for stop-without-reasons (DB-backed)            #
# ------------------------------------------------------------------ #


def _load_annotation_index(conn, segment_id: int) -> np.ndarray:
    """Load annotated frame indices for a segment from the DB.

    Returns a sorted int array of frame indices that have annotations,
    or an empty array if no annotations exist.
    """
    rows = conn.execute(
        "SELECT frame_index FROM frame_annotations "
        "WHERE segment_id = ? ORDER BY frame_index",
        (segment_id,),
    ).fetchall()
    if not rows:
        return np.array([], dtype=np.int64)
    return np.array([r["frame_index"] for r in rows], dtype=np.int64)


def _load_justified_frames(
    conn,
    segment_id: int,
    cfg: FilterConfig,
) -> set[int]:
    """Load frame indices that have a stop-justification (crosswalk or pedestrians).

    Returns the set of annotated frame indices where either:
    - at least one crosswalk detection has confidence >= threshold, or
    - at least ``min_pedestrians`` detections have confidence >= threshold.

    The query deliberately omits the confidence/class filter from SQL to force
    SQLite to drive the JOIN from ``frame_annotations.segment_id`` (tens of
    rows) rather than scanning the 30M-row ``detections`` table via the
    ``(class_label, confidence)`` index.  Filtering is done in Python on the
    small result set (~100-200 rows per segment).
    """
    rows = conn.execute(
        """SELECT fa.frame_index, d.class_label, d.confidence
           FROM frame_annotations fa
           JOIN detections d ON fa.frame_ann_id = d.frame_ann_id
           WHERE fa.segment_id = ?""",
        (segment_id,),
    ).fetchall()

    if not rows:
        return set()

    # Filter by confidence in Python (trivial cost on ~200 rows)
    justified: set[int] = set()
    frame_peds: dict[int, int] = {}
    for r in rows:
        fi = r["frame_index"]
        if r["class_label"] == "crosswalk" and r["confidence"] >= cfg.crosswalk_confidence:
            justified.add(fi)
        elif r["class_label"] == "pedestrian" and r["confidence"] >= cfg.pedestrian_confidence:
            frame_peds[fi] = frame_peds.get(fi, 0) + 1

    for fi, cnt in frame_peds.items():
        if cnt >= cfg.min_pedestrians:
            justified.add(fi)

    return justified


# ------------------------------------------------------------------ #
#  Main filter entry point                                            #
# ------------------------------------------------------------------ #


def compute_all_metrics(
    pose: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all per-frame metrics for a single segment.

    Computes scipy Rotation (as_matrix, as_euler) once and reuses across all
    orientation-dependent metrics to avoid redundant work.

    Returns a dict with keys matching the ``segment_filter_data`` columns.
    """
    # Compute rotation representations once (the expensive part)
    quats = pose[:, 3:]
    rotations = R.from_quat(quats)
    rot_matrices = rotations.as_matrix()  # (N, 3, 3)
    euler = rotations.as_euler("ZYX", degrees=True)  # (N, 3)

    velocities = compute_velocities(pose)
    forward_camera_angles = compute_forward_camera_angles(
        pose, rot_matrices=rot_matrices,
    )
    roll_changes, pitch_changes, yaw_changes = compute_orientation_changes(
        pose, rot_matrices=rot_matrices, euler=euler,
    )
    abs_pitch, abs_roll = compute_absolute_orientation(
        pose, rot_matrices=rot_matrices, euler=euler,
    )
    height_changes = compute_height_changes(pose)
    return {
        "velocities": velocities.astype(np.float32),
        "forward_camera_angles": forward_camera_angles.astype(np.float32),
        "roll_changes": roll_changes.astype(np.float32),
        "pitch_changes": pitch_changes.astype(np.float32),
        "yaw_changes": yaw_changes.astype(np.float32),
        "abs_pitch": abs_pitch.astype(np.float32),
        "abs_roll": abs_roll.astype(np.float32),
        "height_changes": height_changes.astype(np.float32),
    }


def compute_filter_masks(
    metrics: dict[str, np.ndarray],
    cfg: FilterConfig,
    *,
    conn=None,
    segment_id: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-filter boolean masks (True = passes that filter).

    Metrics are smoothed before thresholding.  The stop-without-reasons
    mask is included only when *conn* and *segment_id* are provided.

    The returned dict's iteration order is the filter order baked into the
    legacy ``build_valid_mask`` AND-chain:

        forward_camera_angle, roll_change, pitch_change, yaw_change,
        abs_pitch, abs_roll, velocity_spike, height_change,
        sustained_slow, stop_without_reasons.
    """
    N = len(metrics["velocities"])
    hw = cfg.avg_window
    masks: dict[str, np.ndarray] = {}

    # 1) Forward-vs-camera angle
    s = smooth_window(metrics["forward_camera_angles"], hw)
    masks["forward_camera_angle"] = s <= cfg.forward_camera_max_angle

    # 2) Roll changes
    s = smooth_window(metrics["roll_changes"], hw)
    masks["roll_change"] = s <= cfg.max_roll_change

    # 3) Pitch changes
    s = smooth_window(metrics["pitch_changes"], hw)
    masks["pitch_change"] = s <= cfg.max_pitch_change

    # 4) Yaw changes
    s = smooth_window(metrics["yaw_changes"], hw)
    masks["yaw_change"] = s <= cfg.max_yaw_change

    # 5) Absolute pitch bounds
    s = smooth_window(metrics["abs_pitch"], hw)
    masks["abs_pitch"] = s <= cfg.max_abs_pitch

    # 6) Absolute roll bounds
    s = smooth_window(metrics["abs_roll"], hw)
    masks["abs_roll"] = s <= cfg.max_abs_roll

    # 7) Velocity spikes (scale-invariant via segment median)
    vel = metrics["velocities"]
    nonzero_vel = vel[vel > 1e-8]
    median_vel = float(np.median(nonzero_vel)) if len(nonzero_vel) > 0 else 0.0
    if median_vel > 0:
        masks["velocity_spike"] = vel <= cfg.velocity_spike_factor * median_vel
    else:
        masks["velocity_spike"] = np.ones(N, dtype=bool)

    # 8) Height changes
    s = smooth_window(metrics["height_changes"], hw)
    masks["height_change"] = s <= cfg.max_height_change_ratio

    # 9) Sustained low speed (flag long runs of slow frames)
    slow_mask = np.ones(N, dtype=bool)
    if median_vel > 0:
        vel_smooth = smooth_window(vel, hw)
        is_slow = vel_smooth < cfg.sustained_slow_factor * median_vel
        changes = np.diff(is_slow.astype(np.int8))
        run_starts = np.where(changes == 1)[0] + 1
        run_ends = np.where(changes == -1)[0] + 1
        if is_slow[0]:
            run_starts = np.concatenate([[0], run_starts])
        if is_slow[-1]:
            run_ends = np.concatenate([run_ends, [N]])
        for a, b in zip(run_starts, run_ends):
            if b - a >= cfg.sustained_slow_frames:
                slow_mask[a:b] = False
    masks["sustained_slow"] = slow_mask

    # 10) Stop-without-reasons (requires annotations in DB)
    if conn is not None and segment_id is not None:
        stop_mask = np.ones(N, dtype=bool)
        ann_indices = _load_annotation_index(conn, segment_id)
        if len(ann_indices) > 0:
            vel_smooth = smooth_window(metrics["velocities"], hw)
            stopped = np.where(vel_smooth < cfg.stop_velocity_threshold)[0]
            if len(stopped) > 0:
                justified_frames = _load_justified_frames(conn, segment_id, cfg)
                if not justified_frames:
                    stop_mask[stopped] = False
                else:
                    for i in stopped:
                        dists = np.abs(ann_indices - i)
                        nearest_pos = np.argmin(dists)
                        nearest_frame = int(ann_indices[nearest_pos])
                        if int(dists[nearest_pos]) > cfg.stop_max_ann_distance:
                            stop_mask[i] = False
                        elif nearest_frame not in justified_frames:
                            stop_mask[i] = False
        masks["stop_without_reasons"] = stop_mask

    return masks


def build_valid_mask(
    metrics: dict[str, np.ndarray],
    cfg: FilterConfig,
    *,
    conn=None,
    segment_id: int | None = None,
) -> np.ndarray:
    """Combined per-frame validity = AND of every filter mask."""
    masks = compute_filter_masks(
        metrics, cfg, conn=conn, segment_id=segment_id,
    )
    return np.logical_and.reduce(list(masks.values()))


_COMMIT_BATCH_SIZE = 500


def run_filters(
    db_path: str,
    cfg: FilterConfig | None = None,
    *,
    segments: list[int] | None = None,
) -> None:
    """Compute filter metrics for segments and store results in the database.

    If *segments* is None, processes all segments.  Otherwise processes only
    the given segment IDs.

    Uses separate read/write connections to avoid WAL-scan slowdown: writes
    to ``segment_filter_data`` grow the WAL, and a shared connection would
    force every subsequent annotation read to scan through it.  A dedicated
    readonly connection bypasses this entirely.

    Poses are batch-loaded per chunk (~170 MB) for efficiency.
    Results are committed every 500 segments.
    """
    if cfg is None:
        cfg = FilterConfig()

    # Separate connections: reader never sees writer's WAL growth.
    read_conn = get_connection(db_path, readonly=True)
    write_conn = get_connection(db_path)
    write_conn.execute("PRAGMA synchronous = NORMAL")

    if segments is None:
        rows = read_conn.execute(
            "SELECT segment_id, pose_path, num_frames FROM segments"
        ).fetchall()
    else:
        placeholders = ",".join("?" * len(segments))
        rows = read_conn.execute(
            f"SELECT segment_id, pose_path, num_frames "
            f"FROM segments WHERE segment_id IN ({placeholders})",
            segments,
        ).fetchall()

    # Process in chunks: batch-load poses to avoid per-segment DB round-trips
    # while keeping memory usage bounded (~170 MB per 5000-segment chunk).
    _CHUNK = 5000
    uncommitted = 0
    pbar = tqdm(total=len(rows), desc="Computing filters")

    for chunk_start in range(0, len(rows), _CHUNK):
        chunk_rows = rows[chunk_start : chunk_start + _CHUNK]
        chunk_ids = [r["segment_id"] for r in chunk_rows]

        # Batch-load poses for this chunk
        ph = ",".join("?" * len(chunk_ids))
        pose_rows = read_conn.execute(
            f"SELECT segment_id, pose_data FROM segment_poses "
            f"WHERE segment_id IN ({ph})",
            chunk_ids,
        ).fetchall()
        pose_cache: dict[int, np.ndarray] = {}
        for pr in pose_rows:
            pose = load_pose_from_blob(pr["pose_data"])
            if pose.shape[0] > 0:
                pose_cache[pr["segment_id"]] = pose

        for row in chunk_rows:
            seg_id = row["segment_id"]

            # Use cached pose; fall back to text file
            pose = pose_cache.get(seg_id)
            if pose is None or pose.shape[0] == 0:
                pose_path = row["pose_path"]
                if pose_path:
                    pose = load_pose_from_text(pose_path)
                else:
                    pbar.update(1)
                    continue
            if pose.shape[0] == 0:
                pbar.update(1)
                continue

            metrics = compute_all_metrics(pose)
            valid_mask = build_valid_mask(
                metrics, cfg, conn=read_conn, segment_id=seg_id,
            )

            write_conn.execute(
                """INSERT OR REPLACE INTO segment_filter_data
                   (segment_id, velocities, forward_camera_angles,
                    roll_changes, pitch_changes, yaw_changes,
                    abs_pitch, abs_roll, height_changes, valid_mask)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    seg_id,
                    metrics["velocities"].tobytes(),
                    metrics["forward_camera_angles"].tobytes(),
                    metrics["roll_changes"].tobytes(),
                    metrics["pitch_changes"].tobytes(),
                    metrics["yaw_changes"].tobytes(),
                    metrics["abs_pitch"].tobytes(),
                    metrics["abs_roll"].tobytes(),
                    metrics["height_changes"].tobytes(),
                    valid_mask.astype(np.uint8).tobytes(),
                ),
            )
            uncommitted += 1
            if uncommitted >= _COMMIT_BATCH_SIZE:
                write_conn.commit()
                uncommitted = 0
            pbar.update(1)

        # Free chunk memory
        del pose_cache

    pbar.close()
    if uncommitted > 0:
        write_conn.commit()
    read_conn.close()
    write_conn.close()
    print(f"Filters computed for {len(rows)} segments")
