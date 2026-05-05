"""Build and cache filtered lookup tables (LUT) for training.

A LUT entry is a ``(segment_local_idx, pose_start)`` pair that identifies one
valid training window.  A window is valid when **every** frame in
``[pose_start, pose_start + context_size + wp_length)`` passes the combined
filter mask (after pose-step subsampling).

The output ``.npz`` file contains:
    segment_names  : (S,) array of segment name strings
    segment_paths  : (S, 4) object array [pose_path, feature_path, rgb_path, ann_dir]
    lut            : (M, 2) int32 array — columns (segment_local_idx, pose_start)
    video_ranges   : (S, 2) int32 array — LUT row range per segment
    filter_cfg     : JSON string of the filter configuration
    pose_step      : int
    context_size   : int
    wp_length      : int
    split          : str

Usage::

    python -m curation.cli build-lut --db dataset.db --split train \\
        --context-size 5 --wp-length 5 --pose-step 2 -o lut_train.npz
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .database import get_connection
from .filters import FilterConfig


def _mask_after_subsample(raw_mask: np.ndarray, pose_step: int) -> np.ndarray:
    """Subsample *raw_mask* by *pose_step*, matching the training pipeline."""
    return raw_mask[::pose_step]


def build_lut(
    db_path: str,
    split: str,
    context_size: int,
    wp_length: int,
    pose_step: int = 1,
    interval: int | None = None,
    filter_cfg: FilterConfig | None = None,
    output_path: str | None = None,
) -> Path:
    """Build a filtered LUT and save it as ``.npz``.

    Parameters
    ----------
    db_path : str
        Path to the curation SQLite database.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    context_size : int
        Number of history frames per sample (observation window).
    wp_length : int
        Number of future waypoint frames per sample.
    pose_step : int
        Subsampling stride applied to pose frames (matches
        ``pose_fps // target_fps`` in the training config).
    interval : int or None
        Stride between consecutive LUT entries within a segment.  Defaults to
        *context_size* (matching the existing ``CarlaFeatDataset`` behaviour).
    filter_cfg : FilterConfig or None
        The same config used in ``run_filters``.  Stored in the ``.npz``
        for reproducibility and cache invalidation.
    output_path : str or None
        Where to write the ``.npz``.  Defaults to
        ``{db_dir}/lut_{split}_cs{context_size}_wp{wp_length}.npz``.

    Returns
    -------
    Path to the written ``.npz`` file.
    """
    if interval is None:
        interval = context_size
    if filter_cfg is None:
        filter_cfg = FilterConfig()

    conn = get_connection(db_path, readonly=True)
    window_size = context_size + wp_length

    # Fetch segments for the requested split that have filter data
    rows = conn.execute(
        """SELECT s.segment_id, s.name, s.pose_path, s.feature_path,
                  s.rgb_path, s.annotation_dir, s.num_frames,
                  f.valid_mask
           FROM segments s
           JOIN segment_filter_data f ON s.segment_id = f.segment_id
           WHERE s.split = ?
           ORDER BY s.name""",
        (split,),
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            f"No segments with filter data found for split='{split}'. "
            "Run `ingest` and `filter` first."
        )

    segment_names: list[str] = []
    segment_paths: list[list[str]] = []
    lut_entries: list[tuple[int, int]] = []
    video_ranges: list[tuple[int, int]] = []

    for row in tqdm(rows, desc=f"Building LUT ({split})"):
        raw_mask = np.frombuffer(row["valid_mask"], dtype=np.uint8).astype(bool)
        mask = _mask_after_subsample(raw_mask, pose_step)
        n_subsampled = len(mask)
        usable = n_subsampled - window_size

        if usable <= 0:
            continue

        seg_local_idx = len(segment_names)
        segment_names.append(row["name"])
        segment_paths.append([
            row["pose_path"] or "",
            row["feature_path"] or "",
            row["rgb_path"] or "",
            row["annotation_dir"] or "",
        ])

        start_lut = len(lut_entries)
        for pose_start in range(0, usable, interval):
            window_slice = mask[pose_start: pose_start + window_size]
            if window_slice.all():
                lut_entries.append((seg_local_idx, pose_start))
        end_lut = len(lut_entries)
        video_ranges.append((start_lut, end_lut))

    if not lut_entries:
        raise RuntimeError(
            f"LUT is empty — all windows filtered out for split='{split}'. "
            "Consider relaxing filter thresholds."
        )

    # Serialise
    lut_arr = np.array(lut_entries, dtype=np.int32)
    ranges_arr = np.array(video_ranges, dtype=np.int32)
    names_arr = np.array(segment_names, dtype=object)
    paths_arr = np.array(segment_paths, dtype=object)

    if output_path is None:
        db_dir = Path(db_path).parent
        output_path = str(
            db_dir / f"lut_{split}_cs{context_size}_wp{wp_length}.npz"
        )

    np.savez(
        output_path,
        segment_names=names_arr,
        segment_paths=paths_arr,
        lut=lut_arr,
        video_ranges=ranges_arr,
        filter_cfg=filter_cfg.to_json(),
        pose_step=pose_step,
        context_size=context_size,
        wp_length=wp_length,
        split=split,
    )

    # Register in DB
    cfg_hash = hashlib.sha256(filter_cfg.to_json().encode()).hexdigest()[:16]
    lut_name = f"{split}_cs{context_size}_wp{wp_length}_ps{pose_step}_{cfg_hash}"
    conn = get_connection(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO lut_cache
           (name, split, context_size, wp_length, pose_step,
            filter_cfg, file_path, num_entries)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (lut_name, split, context_size, wp_length, pose_step,
         filter_cfg.to_json(), output_path, len(lut_entries)),
    )
    conn.commit()
    conn.close()

    total_possible = sum(
        max(0, len(np.frombuffer(r["valid_mask"], dtype=np.uint8)[::pose_step]) - window_size)
        // interval
        for r in rows
        if len(np.frombuffer(r["valid_mask"], dtype=np.uint8)[::pose_step]) > window_size
    )
    pct = 100.0 * len(lut_entries) / total_possible if total_possible else 0.0
    print(
        f"LUT saved: {output_path}\n"
        f"  {len(lut_entries)} valid entries from {len(segment_names)} segments "
        f"({pct:.1f}% of {total_possible} possible windows)"
    )
    return Path(output_path)
