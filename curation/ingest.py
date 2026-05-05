"""Scan a dataset root and populate the curation database.

Usage::

    python -m curation.cli ingest /path/to/youtube_videos --db dataset.db

The directory is expected to contain at least a ``pose/`` subdirectory.
Optional modality directories: ``dino/``, ``rgb/``, ``annotations/``.

Annotation JSONs (``detections.json``, ``crosswalks.json``) are parsed and
materialised into the ``frame_annotations`` and ``detections`` tables so that
downstream queries can filter by count or confidence without re-reading files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .database import create_schema, get_connection

# Regex to split "VideoTitle_0003.txt" -> ("VideoTitle", "0003")
_SEG_RE = re.compile(r"^(.+)_(\d{4})$")


def _parse_segment_name(stem: str) -> tuple[str, int] | None:
    m = _SEG_RE.match(stem)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _load_and_parse_pose(pose_path: str) -> np.ndarray:
    """Parse a pose text file into an (N, 7) float64 array.

    Drops the frame-index column (column 0) and truncates at the first NaN.
    Returns an empty (0, 7) array when the file has no valid rows.
    """
    raw = np.loadtxt(pose_path)
    if raw.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    pose = raw[:, 1:] if raw.shape[1] == 8 else raw
    nan_mask = np.isnan(pose).any(axis=1)
    if nan_mask.any():
        pose = pose[: np.argmax(nan_mask)]
    return pose.astype(np.float64)


def _build_normalized_index(directory: str, extensions: tuple[str, ...]) -> dict[str, str]:
    """Build a quote-stripped filename lookup for a directory.

    Returns {normalized_stem: full_path} where normalised means single-quotes
    removed (handles yt-dlp naming inconsistencies).
    """
    index: dict[str, str] = {}
    if not os.path.isdir(directory):
        return index
    for fn in os.listdir(directory):
        if any(fn.endswith(ext) for ext in extensions):
            stem = os.path.splitext(fn)[0]
            key = stem.replace("'", "")
            index[key] = os.path.join(directory, fn)
    return index


def _build_dir_index(directory: str) -> dict[str, str]:
    """Index sub-directories by quote-normalised name."""
    index: dict[str, str] = {}
    if not os.path.isdir(directory):
        return index
    for fn in os.listdir(directory):
        full = os.path.join(directory, fn)
        if os.path.isdir(full):
            key = fn.replace("'", "")
            index[key] = full
    return index


def _ingest_annotations(cursor, segment_id: int, annotation_dir: str | None) -> int:
    """Parse detections.json and crosswalks.json, insert into DB.

    Returns the number of annotated frames inserted.
    """
    if not annotation_dir or not os.path.isdir(annotation_dir):
        return 0

    ann_path = Path(annotation_dir)

    # Load both annotation files (if they exist)
    ped_data: dict[str, dict] = {}
    det_path = ann_path / "detections.json"
    if det_path.exists():
        with open(det_path) as f:
            ped_data = json.load(f)

    cw_data: dict[str, dict] = {}
    cw_path = ann_path / "crosswalks.json"
    if cw_path.exists():
        with open(cw_path) as f:
            cw_data = json.load(f)

    if not ped_data and not cw_data:
        return 0

    # Union of all annotated frame indices
    all_frames = sorted(set(ped_data.keys()) | set(cw_data.keys()), key=int)
    inserted = 0

    for frame_key in all_frames:
        frame_idx = int(frame_key)

        # Pedestrian data
        p_info = ped_data.get(frame_key, {})
        pedestrians = p_info.get("pedestrians") or p_info.get("detections") or []
        ped_count = len(pedestrians)

        # Crosswalk data
        c_info = cw_data.get(frame_key, {})
        crosswalks = c_info.get("detections") or []
        cw_count = len(crosswalks)

        # Insert frame annotation summary row
        cursor.execute(
            """INSERT OR REPLACE INTO frame_annotations
               (segment_id, frame_index, pedestrian_count, crosswalk_count)
               VALUES (?, ?, ?, ?)""",
            (segment_id, frame_idx, ped_count, cw_count),
        )
        frame_ann_id = cursor.lastrowid

        # Insert individual pedestrian detections
        for det in pedestrians:
            bbox = det.get("bbox", [None, None, None, None])
            cursor.execute(
                """INSERT INTO detections
                   (frame_ann_id, class_label, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?, 'pedestrian', ?, ?, ?, ?, ?)""",
                (frame_ann_id, det.get("confidence", 0.0),
                 bbox[0], bbox[1], bbox[2], bbox[3]),
            )

        # Insert individual crosswalk detections
        for det in crosswalks:
            bbox = det.get("bbox", [None, None, None, None])
            cursor.execute(
                """INSERT INTO detections
                   (frame_ann_id, class_label, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?, 'crosswalk', ?, ?, ?, ?, ?)""",
                (frame_ann_id, det.get("confidence", 0.0),
                 bbox[0], bbox[1], bbox[2], bbox[3]),
            )

        inserted += 1

    return inserted


def ingest(
    data_root: str | Path,
    db_path: str | Path,
    *,
    pose_subdir: str = "pose",
    feature_subdir: str = "dino",
    rgb_subdir: str = "rgb",
    annotation_subdir: str = "annotations",
) -> int:
    """Ingest all segments from *data_root* into the database at *db_path*.

    Every segment is inserted with ``split = NULL``.  Use
    :func:`assign_splits` (after filtering) to assign train/val/test.

    Returns the number of segments inserted.
    """
    data_root = Path(data_root)
    create_schema(db_path)
    conn = get_connection(db_path)

    pose_dir = str(data_root / pose_subdir)
    feature_dir = str(data_root / feature_subdir)
    rgb_dir = str(data_root / rgb_subdir)
    ann_dir = str(data_root / annotation_subdir)

    # Discover segments from pose files (authoritative source)
    pose_files = sorted(
        f for f in os.listdir(pose_dir) if f.endswith(".txt")
    )
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {pose_dir}")

    # Build lookup indices for other modalities (quote-normalised)
    feat_index = _build_normalized_index(feature_dir, (".pt",))
    rgb_index = _build_normalized_index(rgb_dir, (".mp4", ".webm", ".mkv", ".avi"))
    ann_index = _build_dir_index(ann_dir)

    # Group segments by video
    video_segments: dict[str, list[tuple[int, str, str]]] = {}
    for pf in pose_files:
        stem = os.path.splitext(pf)[0]
        parsed = _parse_segment_name(stem)
        if parsed is None:
            continue
        video_name, seg_idx = parsed
        pose_path = os.path.join(pose_dir, pf)
        video_segments.setdefault(video_name, []).append((seg_idx, stem, pose_path))

    inserted = 0
    cursor = conn.cursor()

    for video_name in tqdm(sorted(video_segments), desc="Ingesting videos"):
        # Upsert video
        cursor.execute(
            "INSERT OR IGNORE INTO videos (name) VALUES (?)", (video_name,)
        )
        cursor.execute(
            "SELECT video_id FROM videos WHERE name = ?", (video_name,)
        )
        video_id = cursor.fetchone()["video_id"]

        for seg_idx, stem, pose_path in sorted(video_segments[video_name]):
            norm_stem = stem.replace("'", "")

            # Parse pose file once — get both frame count and binary data
            pose = _load_and_parse_pose(pose_path)
            num_frames = pose.shape[0]
            if num_frames == 0:
                continue

            # Resolve modality paths
            feat_path = feat_index.get(f"{norm_stem}")
            rgb_path = rgb_index.get(f"{norm_stem}")
            ann_path = ann_index.get(f"{norm_stem}")

            cursor.execute(
                """INSERT OR REPLACE INTO segments
                   (video_id, segment_index, name, pose_path, feature_path,
                    rgb_path, annotation_dir, num_frames, split)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                (video_id, seg_idx, stem, pose_path, feat_path,
                 rgb_path, ann_path, num_frames),
            )
            segment_id = cursor.lastrowid

            # Store pre-parsed pose as binary BLOB
            cursor.execute(
                "INSERT OR REPLACE INTO segment_poses (segment_id, pose_data) "
                "VALUES (?, ?)",
                (segment_id, pose.tobytes()),
            )

            # Materialise annotation data into DB tables
            _ingest_annotations(cursor, segment_id, ann_path)

            inserted += 1

        # Commit per-video to keep transaction size manageable
        conn.commit()

    conn.close()
    print(f"Ingested {inserted} segments from {len(video_segments)} videos")
    return inserted


def assign_splits(
    db_path: str | Path,
    *,
    num_train: int,
    num_val: int = 0,
    num_test: int = 0,
    seed: int = 42,
    only_filtered: bool = True,
) -> dict[str, int]:
    """Assign train/val/test splits to segments.

    By default, only segments that have filter data (i.e. have been through
    the ``filter`` step) are considered.  Set *only_filtered* to ``False``
    to assign splits to all segments regardless.

    The candidate segments are shuffled deterministically with *seed*,
    then sliced:

        [0, num_train)                      → train
        [num_train, num_train + num_val)    → val
        [num_train + num_val, ... + num_test) → test

    Segments beyond the requested counts keep ``split = NULL``.

    Returns a dict ``{'train': n, 'val': n, 'test': n, 'unassigned': n}``.
    """
    conn = get_connection(db_path)

    # Select candidate segments
    if only_filtered:
        rows = conn.execute(
            """SELECT s.segment_id FROM segments s
               JOIN segment_filter_data f ON s.segment_id = f.segment_id
               ORDER BY s.segment_id"""
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT segment_id FROM segments ORDER BY segment_id"
        ).fetchall()

    seg_ids = [r["segment_id"] for r in rows]

    # Reset all splits first
    conn.execute("UPDATE segments SET split = NULL")

    # Shuffle deterministically
    rng = np.random.RandomState(seed)
    shuffled = list(seg_ids)
    rng.shuffle(shuffled)

    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}

    for i, sid in enumerate(shuffled):
        if i < num_train:
            split = "train"
        elif i < num_train + num_val:
            split = "val"
        elif i < num_train + num_val + num_test:
            split = "test"
        else:
            counts["unassigned"] += 1
            continue
        conn.execute(
            "UPDATE segments SET split = ? WHERE segment_id = ?",
            (split, sid),
        )
        counts[split] += 1

    conn.commit()
    conn.close()

    total = sum(counts.values())
    print(
        f"Splits assigned ({total} candidates, "
        f"{'filtered only' if only_filtered else 'all segments'}):"
    )
    for k in ("train", "val", "test", "unassigned"):
        print(f"  {k:>10s}: {counts[k]}")
    return counts
