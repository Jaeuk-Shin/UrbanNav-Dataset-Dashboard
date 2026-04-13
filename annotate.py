#!/usr/bin/env python3
"""Annotation pipeline for the OMR outdoor robot dataset.

Supports both JPG-frame directories (rgb/{segment}/*.jpg) and MP4 video
files (rgb/{segment}.mp4).  Use --input-format to select the layout
('jpg' default, or 'video').  For video mode, --fps controls the
subsampling rate (default 1.0 fps).

Subcommands (auto-discovered from stages/):
    embed       SigLIP embeddings for text-based image retrieval
    detect      YOLOv8 pedestrian detection and counting
    caption     Florence-2 detailed captioning + object-detection tags
    segment     Grounding DINO + SAM2 open-vocabulary segmentation masks
    query       Retrieve images by text similarity (uses stored embeddings)
    all         Run embed -> detect -> caption -> segment

Multi-GPU:
    python annotate.py all --num-gpus 4

Video dataset example:
    python annotate.py all --data-root video_dataset --fps 0.2
"""

import argparse

from core.parallel import run_parallel
from stages import STAGES
from stages.query import run_query


# Ordered list of stages for the "all" command
ALL_STAGES = ("embed", "detect", "caption", "segment")


def run_all(args):
    for stage_name in ALL_STAGES:
        print(f"\n{'=' * 60}\n  Stage: {stage_name}\n{'=' * 60}")
        run_parallel(stage_name, args)
    print(f"\n{'=' * 60}\n  All stages complete.\n{'=' * 60}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="dataset")
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--segments", default=None)
    common.add_argument("--num-gpus", type=int, default=1)
    common.add_argument("--subsample", type=int, default=1,
                        help="Keep every Nth frame -- JPG dirs only (default: 1 = all)")
    common.add_argument("--fps", type=float, default=None,
                        help="Video subsampling rate in fps (default: 1.0)")
    common.add_argument("--input-format", choices=["jpg", "video"], default="jpg",
                        help="Input layout: 'jpg' = rgb/{seg}/*.jpg dirs, "
                             "'video' = rgb/{seg}.mp4 files (default: jpg)")

    p = argparse.ArgumentParser(description="OMR dataset annotation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Register stages from the registry
    for name, cls in STAGES.items():
        sp = sub.add_parser(name, parents=[common])
        cls.add_arguments(sp)

    # Query subcommand (not a BaseStage)
    qp = sub.add_parser("query", parents=[common])
    qp.add_argument("text")
    qp.add_argument("--top-k", type=int, default=20)
    qp.add_argument("--save-to", default=None)
    qp.add_argument("--vis-masks", action="store_true",
                     help="Generate colorized mask overlays for saved results")
    qp.add_argument("--vis-detections", action="store_true",
                     help="Generate detection bounding-box overlays for saved results")

    # "all" meta-command — collect args from every stage
    sp_all = sub.add_parser("all", parents=[common])
    for name in ALL_STAGES:
        if name in STAGES:
            STAGES[name].add_arguments(sp_all)

    args = p.parse_args()

    # Set defaults for stage-specific args when not present
    for attr, default in [("categories", None), ("box_threshold", 0.3),
                          ("use_caption_tags", False),
                          ("detect_classes", "person"), ("detect_conf", 0.25)]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    if args.cmd == "query":
        run_query(args)
    elif args.cmd == "all":
        run_all(args)
    else:
        run_parallel(args.cmd, args)


if __name__ == "__main__":
    main()
