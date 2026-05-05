#!/usr/bin/env python3
"""Unified CLI for the dataset curation pipeline.

Usage::

    # 1) Ingest — scan a dataset root and populate the database
    python -m curation ingest /path/to/youtube_videos --db youtube.db

    # 2) Filter — compute quality metrics and per-frame validity masks
    python -m curation filter --db youtube.db

    # 3) Assign splits — among filtered segments only
    python -m curation assign-splits --db youtube.db \\
        --num-train 1400 --num-val 50

    # 4) Build LUT — create a cached lookup table for training
    python -m curation build-lut --db youtube.db --split train \\
        --context-size 5 --wp-length 5 --pose-step 2

    # 5) Stats — show summary statistics
    python -m curation stats --db youtube.db
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def cmd_ingest(args):
    from .ingest import ingest

    ingest(
        data_root=args.data_root,
        db_path=args.db,
        pose_subdir=args.pose_subdir,
        feature_subdir=args.feature_subdir,
        rgb_subdir=args.rgb_subdir,
        annotation_subdir=args.annotation_subdir,
    )


def cmd_assign_splits(args):
    from .ingest import assign_splits

    assign_splits(
        db_path=args.db,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
        only_filtered=not args.all_segments,
    )


def cmd_filter(args):
    from .filters import FilterConfig, run_filters

    cfg = FilterConfig(
        forward_camera_max_angle=args.forward_camera_max_angle,
        max_roll_change=args.max_roll_change,
        max_pitch_change=args.max_pitch_change,
        max_yaw_change=args.max_yaw_change,
        max_abs_pitch=args.max_abs_pitch,
        max_abs_roll=args.max_abs_roll,
        velocity_spike_factor=args.velocity_spike_factor,
        max_height_change_ratio=args.max_height_change_ratio,
        sustained_slow_frames=args.sustained_slow_frames,
        sustained_slow_factor=args.sustained_slow_factor,
        stop_velocity_threshold=args.stop_velocity_threshold,
        pedestrian_confidence=args.pedestrian_confidence,
        min_pedestrians=args.min_pedestrians,
        crosswalk_confidence=args.crosswalk_confidence,
        stop_max_ann_distance=args.stop_max_ann_distance,
        avg_window=args.avg_window,
    )
    print(f"Filter config: {cfg.to_json()}")
    run_filters(args.db, cfg)


def cmd_build_lut(args):
    from .build_lut import build_lut
    from .filters import FilterConfig

    cfg = None
    if args.filter_cfg:
        cfg = FilterConfig.from_json(args.filter_cfg)

    build_lut(
        db_path=args.db,
        split=args.split,
        context_size=args.context_size,
        wp_length=args.wp_length,
        pose_step=args.pose_step,
        interval=args.interval,
        filter_cfg=cfg,
        output_path=args.output,
    )


def cmd_stats(args):
    from .database import get_connection

    conn = get_connection(args.db, readonly=True)

    # Segment counts by split
    rows = conn.execute(
        "SELECT split, COUNT(*) as cnt FROM segments GROUP BY split"
    ).fetchall()
    print("Segments by split:")
    for r in rows:
        print(f"  {r['split'] or '(none)':>8s}: {r['cnt']}")

    # Filter coverage
    total = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    filtered = conn.execute("SELECT COUNT(*) FROM segment_filter_data").fetchone()[0]
    print(f"\nFilter coverage: {filtered}/{total} segments")

    if filtered > 0:
        # Compute validity statistics
        frows = conn.execute(
            "SELECT s.split, f.valid_mask, s.num_frames "
            "FROM segment_filter_data f "
            "JOIN segments s ON f.segment_id = s.segment_id"
        ).fetchall()

        by_split: dict[str, dict] = {}
        for r in frows:
            split = r["split"] or "(none)"
            mask = np.frombuffer(r["valid_mask"], dtype=np.uint8).astype(bool)
            info = by_split.setdefault(split, {"total": 0, "valid": 0, "segments": 0})
            info["total"] += len(mask)
            info["valid"] += mask.sum()
            info["segments"] += 1

        print("\nPer-frame validity by split:")
        for split, info in sorted(by_split.items()):
            pct = 100.0 * info["valid"] / info["total"] if info["total"] else 0
            print(
                f"  {split:>8s}: {info['valid']:>10d}/{info['total']:>10d} "
                f"frames valid ({pct:.1f}%) across {info['segments']} segments"
            )

    # LUT cache entries
    luts = conn.execute(
        "SELECT name, split, num_entries, file_path, created_at FROM lut_cache"
    ).fetchall()
    if luts:
        print("\nCached LUTs:")
        for r in luts:
            print(f"  {r['name']}: {r['num_entries']} entries ({r['split']}) — {r['file_path']}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        prog="curation",
        description="Dataset curation pipeline: ingest, filter, build LUT.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    p_ingest = sub.add_parser(
        "ingest", help="Scan dataset root and populate DB (no splits)")
    p_ingest.add_argument("data_root", help="Path to dataset (e.g. youtube_videos)")
    p_ingest.add_argument("--db", required=True, help="SQLite database path")
    p_ingest.add_argument("--pose-subdir", default="pose")
    p_ingest.add_argument("--feature-subdir", default="dino")
    p_ingest.add_argument("--rgb-subdir", default="rgb")
    p_ingest.add_argument("--annotation-subdir", default="annotations")

    # --- assign-splits ---
    p_split = sub.add_parser(
        "assign-splits",
        help="Assign train/val/test splits (run after filter)")
    p_split.add_argument("--db", required=True, help="SQLite database path")
    p_split.add_argument("--num-train", type=int, required=True)
    p_split.add_argument("--num-val", type=int, default=0)
    p_split.add_argument("--num-test", type=int, default=0)
    p_split.add_argument("--seed", type=int, default=42,
                         help="Random seed for shuffling before split")
    p_split.add_argument("--all-segments", action="store_true",
                         help="Assign from all segments, not just filtered ones")

    # --- filter ---
    p_filter = sub.add_parser("filter", help="Compute quality filters")
    p_filter.add_argument("--db", required=True, help="SQLite database path")
    # Orientation change thresholds
    p_filter.add_argument("--forward-camera-max-angle", type=float, default=60.0,
                          help="Max velocity-vs-camera angle in degrees [0,180]")
    p_filter.add_argument("--max-roll-change", type=float, default=30.0,
                          help="Max per-frame roll change (deg)")
    p_filter.add_argument("--max-pitch-change", type=float, default=20.0,
                          help="Max per-frame pitch change (deg)")
    p_filter.add_argument("--max-yaw-change", type=float, default=45.0,
                          help="Max per-frame yaw change (deg)")
    # Absolute orientation bounds
    p_filter.add_argument("--max-abs-pitch", type=float, default=30.0,
                          help="Max absolute pitch from horizontal (deg)")
    p_filter.add_argument("--max-abs-roll", type=float, default=20.0,
                          help="Max absolute roll from level (deg)")
    # Velocity spikes
    p_filter.add_argument("--velocity-spike-factor", type=float, default=10.0,
                          help="Spike threshold as multiple of segment median velocity")
    # Height changes
    p_filter.add_argument("--max-height-change-ratio", type=float, default=0.3,
                          help="Max |Δy|/|Δxyz| per frame (0=flat, 1=vertical)")
    # Sustained low speed
    p_filter.add_argument("--sustained-slow-frames", type=int, default=20,
                          help="Min consecutive slow frames to filter")
    p_filter.add_argument("--sustained-slow-factor", type=float, default=0.15,
                          help="Slow threshold as fraction of segment median velocity")
    # Stop-without-reasons
    p_filter.add_argument("--stop-velocity-threshold", type=float, default=0.005,
                          help="Velocity below which stop-check triggers")
    p_filter.add_argument("--pedestrian-confidence", type=float, default=0.4)
    p_filter.add_argument("--min-pedestrians", type=int, default=1)
    p_filter.add_argument("--crosswalk-confidence", type=float, default=0.3)
    p_filter.add_argument("--stop-max-ann-distance", type=int, default=10,
                          help="Max pose-frame distance to nearest annotation for stop justification")
    # Shared
    p_filter.add_argument("--avg-window", type=int, default=5,
                          help="Sliding window half-size for metric smoothing")

    # --- build-lut ---
    p_lut = sub.add_parser("build-lut", help="Build a cached lookup table")
    p_lut.add_argument("--db", required=True, help="SQLite database path")
    p_lut.add_argument("--split", required=True, choices=["train", "val", "test"])
    p_lut.add_argument("--context-size", type=int, required=True)
    p_lut.add_argument("--wp-length", type=int, required=True)
    p_lut.add_argument("--pose-step", type=int, default=2)
    p_lut.add_argument("--interval", type=int, default=None,
                       help="LUT entry stride (default: context_size)")
    p_lut.add_argument("--filter-cfg", type=str, default=None,
                       help="JSON string of filter config (for cache key)")
    p_lut.add_argument("-o", "--output", type=str, default=None,
                       help="Output .npz path")

    # --- stats ---
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("--db", required=True, help="SQLite database path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {
        "ingest": cmd_ingest,
        "assign-splits": cmd_assign_splits,
        "filter": cmd_filter,
        "build-lut": cmd_build_lut,
        "stats": cmd_stats,
    }[args.command](args)


if __name__ == "__main__":
    main()
