"""Text-based image retrieval using stored SigLIP embeddings."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.discovery import discover_segments
from pipeline.frames import load_frame


def _save_mask_overlay(img, ann_root, seg, frame, save_dir, rank) -> bool:
    """Blend the SAM label map for ``(seg, frame)`` over *img* and write JPEG.

    Returns True iff a mask file existed (regardless of whether it had labels).
    """
    mask_file = ann_root / seg / "masks" / f"{frame}.png"
    if not mask_file.exists():
        return False
    label_map = np.array(Image.open(mask_file))
    n_labels = int(label_map.max())
    if n_labels > 0:
        rng = np.random.RandomState(42)
        pal = np.zeros((n_labels + 1, 3), dtype=np.uint8)
        pal[1:] = rng.randint(60, 255, size=(n_labels, 3))
        blended = Image.blend(img, Image.fromarray(pal[label_map]), alpha=0.4)
    else:
        blended = img
    blended.save(save_dir / f"{rank:03d}_{seg}_{frame}_mask.jpg", quality=90)
    return True


def _save_detection_overlay(img, ann_root, seg, frame, save_dir, rank,
                             font) -> bool:
    """Draw pedestrian boxes for ``(seg, frame)`` over *img* and write JPEG."""
    from PIL import ImageDraw

    det_file = ann_root / seg / "detections.json"
    if not det_file.exists():
        return False
    d = json.loads(det_file.read_text())
    if frame not in d or d[frame]["pedestrian_count"] == 0:
        return False

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for ped in d[frame]["pedestrians"]:
        x1, y1, x2, y2 = ped["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        lbl = f"{ped['confidence']:.2f}"
        tx, ty = x1, y1 - 22
        if ty < 0:
            ty = y2 + 2
        bb = draw.textbbox((tx, ty), lbl, font=font)
        draw.rectangle(bb, fill="lime")
        draw.text((tx, ty), lbl, fill="black", font=font)
    draw_img.save(save_dir / f"{rank:03d}_{seg}_{frame}_det.jpg", quality=90)
    return True


def run_query(args):
    import open_clip

    data_root = Path(args.data_root)
    ann_root = data_root / "annotations"
    device = torch.device(args.device)
    top_k = args.top_k

    meta_file = ann_root / "embedding_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        model_name = meta.get("model", "ViT-SO400M-14-SigLIP-384")
        pretrained = meta.get("pretrained", "webli")
    else:
        model_name = "ViT-SO400M-14-SigLIP-384"
        pretrained = "webli"

    print(f"Loading {model_name} for text encoding ...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    tokens = tokenizer([args.text]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().float().numpy()[0]

    results = []
    segments = discover_segments(args.data_root, fps=getattr(args, "fps", None), input_format=args.input_format)
    for seg, paths in segments.items():
        emb_file = ann_root / seg / "embeddings.npy"
        if not emb_file.exists():
            continue
        embs = np.load(emb_file).astype(np.float32)
        sims = embs @ text_feat
        for ref, sim in zip(paths, sims):
            results.append((float(sim), seg, ref.stem, ref))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f'\nTop {top_k} results for: "{args.text}"\n')
    header = f"{'Rank':<5} {'Score':<7} {'Segment':<40} {'Frame':<8} {'Ped.':<5} Source"
    print(header)
    print("-" * len(header))
    for rank, (sim, seg, frame, ref) in enumerate(results[:top_k], 1):
        ped = "\u2014"
        det_file = ann_root / seg / "detections.json"
        if det_file.exists():
            d = json.loads(det_file.read_text())
            if frame in d:
                ped = str(d[frame]["pedestrian_count"])
        src = str(ref) if isinstance(ref, Path) else f"{ref.video_path}@{ref.stem}"
        print(f"{rank:<5} {sim:<7.3f} {seg:<40} {frame:<8} {ped:<5} {src}")

    if args.save_to:
        save_dir = Path(args.save_to)
        save_dir.mkdir(parents=True, exist_ok=True)

        vis_masks = getattr(args, "vis_masks", False)
        vis_det = getattr(args, "vis_detections", False)

        det_font = None
        if vis_det:
            from PIL import ImageFont
            try:
                det_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
                )
            except OSError:
                det_font = ImageFont.load_default()

        mask_count = 0
        det_count = 0
        for rank, (sim, seg, frame, ref) in enumerate(results[:top_k], 1):
            dst = save_dir / f"{rank:03d}_{seg}_{frame}.jpg"
            if not dst.exists():
                if isinstance(ref, Path):
                    os.symlink(ref.resolve(), dst)
                else:
                    load_frame(ref).save(dst)

            if not (vis_masks or vis_det):
                continue
            img = load_frame(ref)
            if img is None:
                continue

            if vis_masks and _save_mask_overlay(
                img, ann_root, seg, frame, save_dir, rank,
            ):
                mask_count += 1
            if vis_det and _save_detection_overlay(
                img, ann_root, seg, frame, save_dir, rank, det_font,
            ):
                det_count += 1

        parts = []
        if vis_masks and mask_count:
            parts.append(f"{mask_count} mask overlays")
        if vis_det and det_count:
            parts.append(f"{det_count} detection overlays")
        msg = f"\nSaved top-{top_k} -> {save_dir}/"
        if parts:
            msg += f" ({', '.join(parts)})"
        print(msg)
